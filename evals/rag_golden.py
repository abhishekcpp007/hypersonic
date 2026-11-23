"""Score keyword retrieval on the rag-harness golden set.

    uv run --with pyyaml python evals/rag_golden.py [path to rag-harness]
"""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from statistics import mean

import yaml

from hypersonic.rag.ingest import DOCS, index_text
from hypersonic.rag.retrieve import retrieve
from hypersonic.rag.store import Store

DEFAULT_HARNESS = Path.home() / "JobSearch" / "rag-harness"
SCOPE = "golden"
CUTOFFS = (1, 3, 5)
THRESHOLDS = (0.2, 0.25, 0.3, 0.34, 0.4, 0.45, 0.5)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    return next((rank for rank, item in enumerate(ranked, start=1) if item in gold), None)


def summarize(ranks: list[int | None]) -> dict[str, float]:
    scores = {f"recall@{cut}": mean(1.0 if rank and rank <= cut else 0.0 for rank in ranks) for cut in CUTOFFS}
    scores["mrr"] = mean(1 / rank if rank else 0.0 for rank in ranks)
    return {name: round(value, 4) for name, value in scores.items()}


def main() -> None:
    harness = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HARNESS
    golden = yaml.safe_load((harness / "eval" / "golden.yaml").read_text())
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder) / "index.db")
        for doc in sorted((harness / "corpus").glob("*.md")):
            index_text(store, SCOPE, DOCS, doc.stem, doc.read_text(), stamp="eval", headings=True)
        with closing(store.connect()) as db:
            rows = db.execute("SELECT rowid, path FROM chunks ORDER BY rowid").fetchall()
        chunk_ids: dict[int, str] = {}
        counts: dict[str, int] = {}
        for rowid, path in rows:
            chunk_ids[rowid] = f"{path}#{counts.get(path, 0)}"
            counts[path] = counts.get(path, 0) + 1

        labelled = {chunk for item in golden for chunk in item.get("gold_chunks", [])}
        missing = sorted(labelled - set(chunk_ids.values()))
        doc_ranks, chunk_ranks, by_kind = [], [], {}
        confidences = {"answerable": [], "unanswerable": []}
        for item in golden:
            found = retrieve(store, item["question"], [SCOPE])
            if not item["gold_docs"]:
                confidences["unanswerable"].append(found.confidence)
                continue
            confidences["answerable"].append(found.confidence)
            docs = list(dict.fromkeys(hit.path for hit in found.hits))
            chunks = [chunk_ids[hit.rowid] for hit in found.hits]
            doc_ranks.append(first_rank(docs, set(item["gold_docs"])))
            chunk_ranks.append(first_rank(chunks, set(item["gold_chunks"])))
            by_kind.setdefault(item["kind"], []).append(chunk_ranks[-1])

    report = {
        "chunks_indexed": len(rows),
        "gold_chunks_missing_from_index": missing,
        "doc": summarize(doc_ranks),
        "chunk": summarize(chunk_ranks),
        "chunk_mrr_by_kind": {kind: summarize(ranks)["mrr"] for kind, ranks in sorted(by_kind.items())},
        "abstain_sweep": {
            str(cut): {
                "answerable_refused": round(mean(score < cut for score in confidences["answerable"]), 4),
                "unanswerable_refused": round(mean(score < cut for score in confidences["unanswerable"]), 4),
            }
            for cut in THRESHOLDS
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
