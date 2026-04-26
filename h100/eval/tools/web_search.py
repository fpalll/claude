"""
Drop-in replacement for BFCL's SerpAPI-backed web_search.py.
Uses duckduckgo-search (free, no API key) + requests for URL fetching.

Matches the official BFCL function signatures exactly:
  - duckduckgo_search(keywords, max_results, region) -> list[dict]
  - fetch_url_content(url, mode) -> str
"""
import time
import random
import requests
from typing import Optional

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import html2text as _html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False

# Matches the official BFCL browser-mimicking headers
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MAX_CONTENT_CHARS = 8000   # truncate fetched pages to keep token budget sane
_FETCH_TIMEOUT = 20         # seconds, matches official implementation


def duckduckgo_search(
    keywords: str,
    max_results: Optional[int] = 10,
    region: Optional[str] = "wt-wt",
) -> list[dict]:
    """
    Search DuckDuckGo. Returns list of {title, href, body} dicts.
    Retries on rate-limit (matching BFCL's exponential backoff behaviour).
    """
    from duckduckgo_search import DDGS

    backoff = 5
    for attempt in range(4):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    keywords,
                    max_results=max_results,
                    region=region,
                ))
            # Normalise field names to match what the model expects
            return [
                {
                    "title":   r.get("title", ""),
                    "href":    r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except Exception as e:
            err = str(e).lower()
            if "ratelimit" in err or "202" in err or "429" in err:
                wait = min(backoff + random.uniform(0, 2), 120)
                print(f"  [ddg rate-limit] sleeping {wait:.0f}s …")
                time.sleep(wait)
                backoff *= 2
            else:
                return [{"error": str(e), "title": "", "href": "", "snippet": ""}]
    return [{"error": "rate-limit after retries", "title": "", "href": "", "snippet": ""}]


def fetch_url_content(url: str, mode: str = "markdown") -> str:
    """
    Fetch page content.
      mode='raw'      → raw HTML
      mode='markdown' → HTML converted to markdown (most useful for the model)
      mode='truncate' → plain text, scripts/styles stripped
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        html = resp.text

        if mode == "raw":
            return html[:_MAX_CONTENT_CHARS]

        if mode == "markdown":
            if HTML2TEXT_AVAILABLE:
                h = _html2text.HTML2Text()
                h.ignore_links = False
                h.body_width = 0
                return h.handle(html)[:_MAX_CONTENT_CHARS]
            # Fallback if html2text not installed
            mode = "truncate"

        # mode == "truncate"
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        else:
            # Very basic fallback: strip tags with regex
            import re
            text = re.sub(r"<[^>]+>", " ", html)
        return " ".join(text.split())[:_MAX_CONTENT_CHARS]

    except requests.exceptions.Timeout:
        return f"[Error] Timeout fetching {url}"
    except requests.exceptions.HTTPError as e:
        return f"[Error] HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"[Error] {e}"


# OpenAI tool definitions — passed directly to client.chat.completions.create(tools=...)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": (
                "Search the web using DuckDuckGo. "
                "Returns a list of results with title, URL, and snippet. "
                "Use SHORT queries (3-6 words) for best retrieval quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Short search query (3-6 words recommended)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                        "default": 10,
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code, e.g. 'wt-wt' for worldwide (default)",
                        "default": "wt-wt",
                    },
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": (
                "Fetch the full content of a URL. "
                "Always call this after duckduckgo_search to read the actual page — "
                "do NOT rely on snippets alone. "
                "Use mode='markdown' for readable structured text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["raw", "markdown", "truncate"],
                        "description": "Content format: markdown (default), truncate (plain text), raw (HTML)",
                        "default": "markdown",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call and return the result as a string."""
    if name == "duckduckgo_search":
        results = duckduckgo_search(**arguments)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            if "error" in r:
                lines.append(f"[Search error: {r['error']}]")
            else:
                lines.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['snippet']}")
        return "\n\n".join(lines)

    if name == "fetch_url_content":
        return fetch_url_content(**arguments)

    return f"[Error] Unknown tool: {name}"
