import stat
import subprocess
from pathlib import Path

from hypersonic.providers.base import OWNER_ONLY
from hypersonic.rag.chunk import OVERSIZE, WORDS, chunk
from hypersonic.rag.ingest import CODE, DOCS, index_folder, index_text
from hypersonic.rag.retrieve import ATTACH_TOKENS, GATE, attach, pack, retrieve
from hypersonic.rag.tokens import tokenize
from hypersonic.context.squeeze import CHARS_PER_TOKEN

SCOPE = "project"
HANDBOOK = """# Payments handbook

## Refunds

Refunds settle within five business days. A partial refund keeps the original charge open.

## Error codes

NW-4021 means the settlement batch was rejected by the acquirer. Retry after the next batch window.

```
curl -X POST /v1/charges -d amount=100
```
"""


def test_tokenizer_splits_identifiers_and_drops_stop_words():
    assert tokenize("NW-4021") == ["nw-4021", "nw", "4021"]
    assert tokenize("getUserName") == ["getusername", "get", "user", "name"]
    assert tokenize("What is the refund policy?") == ["refund", "policy"]
    assert tokenize("batches runs class") == ["batch", "run", "class"]
    assert tokenize("error 4021.") == ["error", "4021"]
    assert tokenize("café résumé") == ["café", "résumé"]


def test_chunks_carry_their_heading_path_and_keep_fences_whole():
    chunks = chunk(HANDBOOK)
    assert [item.heading for item in chunks] == [
        "Payments handbook > Refunds",
        "Payments handbook > Error codes",
    ]
    assert chunks[0].text.startswith("Refunds settle")
    assert (chunks[0].first_line, chunks[0].last_line) == (5, 5)
    assert "curl -X POST" in chunks[1].text
    assert [item.ordinal for item in chunks] == [0, 1]


def test_headings_can_be_switched_off_for_code_and_logs():
    chunks = chunk("# not a heading\nx = 1\n\n# another comment\ny = 2\n", headings=False)
    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert "# not a heading" in chunks[0].text


def test_oversized_blocks_are_sliced():
    wall = "\n".join(f"line {number} " + "word " * 20 for number in range(200))
    chunks = chunk(wall, headings=False)
    assert len(chunks) > 1
    assert max(len(item.text.split()) for item in chunks) <= OVERSIZE * WORDS
    assert chunks[0].first_line == 1 and chunks[-1].last_line == 200


def test_store_filters_by_scope_and_collection(store):
    index_text(store, SCOPE, DOCS, "/docs/handbook.md", HANDBOOK, "v1", headings=True)
    index_text(store, "other", DOCS, "/else/notes.md", "Refunds are never allowed here.", "v1", headings=True)
    index_text(store, SCOPE, CODE, "/src/refund.py", "def refund(charge): return charge", "v1", headings=False)

    everything = retrieve(store, "refund", [SCOPE])
    assert {hit.path for hit in everything.hits} == {"/docs/handbook.md", "/src/refund.py"}
    only_docs = retrieve(store, "refund", [SCOPE], [DOCS])
    assert [hit.path for hit in only_docs.hits] == ["/docs/handbook.md"]
    assert retrieve(store, "refund", ["nobody"]).hits == []
    assert retrieve(store, "the of and", [SCOPE]).hits == []


def test_identifier_lookup_and_citation_lines(store):
    index_text(store, SCOPE, DOCS, "/docs/handbook.md", HANDBOOK, "v1", headings=True)
    top = retrieve(store, "what does NW-4021 mean", [SCOPE]).hits[0]
    assert top.heading == "Payments handbook > Error codes"
    assert (top.first_line, top.last_line) == (9, 13)
    assert top.score > 0


def test_reindexing_replaces_old_chunks_and_remove_clears_them(store):
    index_text(store, SCOPE, DOCS, "/a.md", "Walrus tusks are long.", "v1", headings=True)
    index_text(store, SCOPE, DOCS, "/a.md", "Penguins huddle for warmth.", "v2", headings=True)
    assert retrieve(store, "walrus", [SCOPE]).hits == []
    assert len(retrieve(store, "penguins", [SCOPE]).hits) == 1
    assert store.stamp(SCOPE, DOCS, "/a.md") == "v2"
    store.remove(SCOPE, DOCS, "/a.md")
    assert retrieve(store, "penguins", [SCOPE]).hits == []
    assert store.paths(SCOPE, DOCS) == []


def test_index_folder_tracks_changes(store, tmp_path):
    docs = tmp_path / "docs"
    (docs / "node_modules").mkdir(parents=True)
    (docs / "a.md").write_text("Walrus tusks are long.")
    (docs / "b.txt").write_text("Penguins huddle for warmth.")
    (docs / "blob.bin").write_bytes(b"\x00\x01")
    (docs / "node_modules" / "junk.md").write_text("walrus junk")
    (docs / ".hidden").mkdir()
    (docs / ".hidden" / "secret.md").write_text("walrus secret")

    assert index_folder(store, SCOPE, DOCS, docs) == {"indexed": 2, "unchanged": 0, "removed": 0}
    assert index_folder(store, SCOPE, DOCS, docs) == {"indexed": 0, "unchanged": 2, "removed": 0}
    assert [hit.path for hit in retrieve(store, "walrus", [SCOPE]).hits] == [str(docs.resolve() / "a.md")]

    (docs / "a.md").write_text("Walrus tusks are long. They also have whiskers.")
    (docs / "b.txt").unlink()
    assert index_folder(store, SCOPE, DOCS, docs) == {"indexed": 1, "unchanged": 0, "removed": 1}
    assert retrieve(store, "penguins", [SCOPE]).hits == []
    assert len(retrieve(store, "whiskers", [SCOPE]).hits) == 1


def test_a_link_that_leaves_the_folder_is_never_indexed(store, tmp_path):
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "keys.md").write_text("walrus password")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "real.md").write_text("Walrus tusks are long.")
    (docs / "inside.md").symlink_to(docs / "real.md")
    (docs / "leak.md").symlink_to(outside / "keys.md")
    (docs / "more").symlink_to(outside, target_is_directory=True)
    index_folder(store, SCOPE, DOCS, docs)
    found = {hit.path for hit in retrieve(store, "walrus", [SCOPE]).hits}
    assert found == {str(docs.resolve() / "real.md"), str(docs.resolve() / "inside.md")}
    assert retrieve(store, "password", [SCOPE]).hits == []


def test_a_git_folder_is_walked_from_anywhere_inside_it(store, tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "inner").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    (repo / "src" / "build").write_text("walrus build notes")
    (repo / "src" / "a.py").write_text("walrus = 1")
    (repo / "src" / "ignored.log").write_text("walrus log")
    (repo / "node_modules" / "junk.md").write_text("walrus junk")
    (repo / ".gitignore").write_text("*.log\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src" / "inner" / "b.py").write_text("walrus = 2")
    subprocess.run(["git", "init", "-q", str(repo / "src" / "inner")], check=True)
    index_folder(store, SCOPE, CODE, repo / "src")
    names = {Path(hit.path).name for hit in retrieve(store, "walrus", [SCOPE]).hits}
    assert names == {"build", "a.py", "b.py"}


def test_the_index_file_is_private(store):
    assert stat.S_IMODE(store.path.stat().st_mode) == OWNER_ONLY


def test_a_passage_indexed_again_is_not_attached_twice(store):
    index_text(store, SCOPE, DOCS, "/a.md", "penguin " + "filler " * 40, "v1", headings=True)
    seen: set[int] = set()
    assert pack(retrieve(store, "penguin", [SCOPE]).hits, 600, seen)
    index_text(store, SCOPE, DOCS, "/a.md", "penguin " + "filler " * 40, "v2", headings=True)
    again = retrieve(store, "penguin", [SCOPE]).hits
    assert pack(again, 600, seen) == ""
    assert pack(again + again, 600, set(), fresh_only=False).count("penguin") == 1


def test_confidence_separates_matches_from_noise(store):
    index_text(store, SCOPE, DOCS, "/docs/handbook.md", HANDBOOK, "v1", headings=True)
    good = retrieve(store, "how long do refunds take to settle", [SCOPE])
    noise = retrieve(store, "who is the chief executive of the walrus company", [SCOPE])
    assert good.confidence > GATE > noise.confidence


def test_pack_respects_budget_and_skips_seen(store):
    for number in range(5):
        index_text(store, SCOPE, DOCS, f"/doc{number}.md", "penguin " + "filler " * 300, "v1", headings=True)
    hits = retrieve(store, "penguin", [SCOPE]).hits
    seen: set[int] = set()
    first = pack(hits, 600, seen)
    assert first.startswith("[docs] /doc")
    assert len(first) <= 600 * CHARS_PER_TOKEN + 200
    assert seen
    second = pack(hits, 600, seen)
    assert second and not set(first.split("\n")[0].split()) == set(second.split("\n")[0].split())
    assert pack(hits, 600, seen, fresh_only=False).split("\n")[0] == first.split("\n")[0]


def test_attach_needs_a_real_question_and_never_repeats(store):
    index_text(store, SCOPE, DOCS, "/docs/handbook.md", HANDBOOK, "v1", headings=True)
    seen: set[int] = set()
    assert attach(store, "thanks", [SCOPE], seen) == ""
    assert attach(store, "who is the chief executive of the walrus company", [SCOPE], seen) == ""
    found = attach(store, "how long do refunds take to settle", [SCOPE], seen)
    assert found.startswith("<context>\n[docs] /docs/handbook.md:5-5 (Payments handbook > Refunds)")
    assert len(found) <= ATTACH_TOKENS * CHARS_PER_TOKEN + 200
    assert "Refunds settle within five business days" in found
    again = attach(store, "how long do refunds take to settle", [SCOPE], seen)
    assert "Refunds settle within five business days" not in again
