"""A stand-in for Ollama's website, serving pages recorded from the real one (D101).

Ollama publishes its library only as pages, so the directory reads pages; these are the real
pages' markup, trimmed to a few models, so the parser is tested against what the site serves and
not against what someone guessed it serves.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

RECORDED = Path(__file__).resolve().parent.parent / "fixtures" / "directory"


class FakeOllamaSite:
    def __init__(self):
        self.pages = {"/library": (RECORDED / "ollama_library.html").read_text()}
        for page in RECORDED.glob("ollama_tags_*.html"):
            name = page.stem.removeprefix("ollama_tags_")
            self.pages[f"/library/{name}/tags"] = page.read_text()
        #: Set to make every page fail, as a site that is down would.
        self.down = False
        self.requests: list[str] = []
        self.app = Starlette(routes=[Route("/{path:path}", self._page, methods=["GET"])])

    async def _page(self, request: Request) -> Response:
        path = request.url.path
        self.requests.append(path)
        if self.down:
            return Response(status_code=503)
        page = self.pages.get(path)
        return HTMLResponse(page) if page is not None else Response(status_code=404)
