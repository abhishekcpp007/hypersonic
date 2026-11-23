from hypersonic.tools import files, search, shell, web
from hypersonic.tools.base import Tool

TOOLS: dict[str, Tool] = {tool.spec.name: tool for tool in (shell.TOOL, files.TOOL, search.TOOL, web.TOOL)}
