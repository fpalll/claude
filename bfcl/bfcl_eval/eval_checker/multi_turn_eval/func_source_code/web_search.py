import os
import random
import time
from typing import Any, Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

# Default mirrors upstream BFCL (SerpAPI). Set WEB_SEARCH_BACKEND to use alternatives.
# See bfcl_eval/.env.example for keys per backend.
WEB_SEARCH_BACKEND_ENV = "WEB_SEARCH_BACKEND"

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


def _region_to_gl_hl(region: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Map BFCL/SerpAPI-style region (e.g. us-en, wt-wt) to Serper-style gl/hl."""
    if not region or region == "wt-wt":
        return None, None
    if "-" not in region:
        return None, None
    gl, hl = region.split("-", 1)
    if len(gl) == 2 and gl.isalpha():
        return gl.lower(), hl.lower() if hl else None
    return None, None


def _normalize_organic_results(
    show_snippet: bool,
    organic: list[dict[str, Any]],
    max_results: Optional[int],
) -> list[dict[str, str]]:
    """
    Convert provider-specific organic rows into BFCL's canonical shape (same as legacy SerpAPI):
    title, href, body (snippet) — matching organic_results from SerpAPI duckduckgo/google.
    Each input row must have keys: title, link, snippet (snippet may be missing).
    """
    cap = max_results if max_results is not None else 10
    cap = min(cap, 50)
    out: list[dict[str, str]] = []
    for result in organic[:cap]:
        title = result.get("title") or ""
        href = result.get("link") or ""
        snippet = result.get("snippet") or ""
        if show_snippet:
            out.append({"title": title, "href": href, "body": snippet})
        else:
            out.append({"title": title, "href": href})
    return out


def _is_rate_limit(exc_or_msg: str) -> bool:
    return "429" in str(exc_or_msg)


class WebSearchAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        # Note: The following two random generators are used to simulate random errors, but that feature is not currently used
        # This one used to determine if we should simulate a random error
        # Outcome (True means simulate error): [True, False, True, True, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, True]
        self._random = random.Random(337)
        # This one is used to determine the content of the error message
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]

    def _run_with_429_retry(self, label: str, fetch_once):
        """fetch_once() -> tuple[success: bool, payload: Any, err_msg: Optional[str]]."""
        backoff = 2
        while True:
            ok, payload, err = fetch_once()
            if ok:
                return payload
            if err and _is_rate_limit(err):
                wait_time = backoff + random.uniform(0, backoff)
                print(
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] 429 from {label}. Retrying in {wait_time:.1f}s…"
                    + "*" * 100
                )
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue
            print(
                "*" * 100
                + f"\n❗️❗️ [WebSearchAPI] Error from {label}: {err}. Not retrying."
                + "*" * 100
            )
            return {"error": err or "unknown error"}

    def _search_serpapi(
        self, keywords: str, max_results: Optional[int], region: Optional[str]
    ):
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        def fetch_once():
            try:
                search = GoogleSearch(params)
                data = search.get_dict()
            except Exception as e:
                err_s = str(e)
                if _is_rate_limit(err_s):
                    return False, None, err_s
                return False, None, err_s

            if "error" in data:
                err_s = str(data["error"])
                if _is_rate_limit(err_s):
                    return False, None, err_s
                return False, None, err_s

            if "organic_results" not in data:
                return (
                    False,
                    None,
                    "Failed to retrieve the search results from server. Please try again later.",
                )

            organic = data["organic_results"]
            organic_norm = []
            for r in organic:
                organic_norm.append(
                    {
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
            return True, organic_norm, None

        raw = self._run_with_429_retry("SerpAPI", fetch_once)
        if isinstance(raw, dict) and "error" in raw:
            return raw
        return _normalize_organic_results(self.show_snippet, raw, max_results)

    def _search_serper(
        self, keywords: str, max_results: Optional[int], region: Optional[str]
    ):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {"error": "SERPER_API_KEY is not set."}

        gl, hl = _region_to_gl_hl(region)
        num = max(1, min(max_results or 10, 100))

        payload: dict[str, Any] = {"q": keywords, "num": num}
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl

        def fetch_once():
            try:
                resp = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 429:
                    return False, None, resp.text
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return False, None, str(e)

            if "organic" not in data:
                return True, [], None
            organic_norm = []
            for r in data["organic"]:
                organic_norm.append(
                    {
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
            return True, organic_norm, None

        raw = self._run_with_429_retry("Serper", fetch_once)
        if isinstance(raw, dict) and "error" in raw:
            return raw
        return _normalize_organic_results(self.show_snippet, raw or [], max_results)

    def _search_tavily(
        self, keywords: str, max_results: Optional[int], region: Optional[str]
    ):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY is not set."}

        top_k = max(1, min(max_results or 10, 20))
        body = {
            "api_key": api_key,
            "query": keywords,
            "max_results": top_k,
            "search_depth": "basic",
        }
        gl, _ = _region_to_gl_hl(region)
        if gl:
            body["country"] = gl.upper()

        def fetch_once():
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json=body,
                    timeout=60,
                )
                if resp.status_code == 429:
                    return False, None, resp.text
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return False, None, str(e)

            rows = data.get("results") or []
            organic_norm = []
            for r in rows:
                organic_norm.append(
                    {
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "snippet": (r.get("content") or "")[:2000],
                    }
                )
            return True, organic_norm, None

        raw = self._run_with_429_retry("Tavily", fetch_once)
        if isinstance(raw, dict) and "error" in raw:
            return raw
        return _normalize_organic_results(self.show_snippet, raw or [], max_results)

    def _search_bing(
        self, keywords: str, max_results: Optional[int], region: Optional[str]
    ):
        key = os.getenv("BING_SEARCH_API_KEY")
        if not key:
            return {"error": "BING_SEARCH_API_KEY is not set."}

        endpoint = os.getenv(
            "BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search"
        )
        count = max(1, min(max_results or 10, 50))
        params = {"q": keywords, "count": count}
        gl, hl = _region_to_gl_hl(region)
        if gl:
            lang = (hl or "en").lower()
            params["mkt"] = f"{lang}-{gl.upper()}"

        def fetch_once():
            try:
                resp = requests.get(
                    endpoint,
                    headers={"Ocp-Apim-Subscription-Key": key},
                    params=params,
                    timeout=60,
                )
                if resp.status_code == 429:
                    return False, None, resp.text
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return False, None, str(e)

            pages = (data.get("webPages") or {}).get("value") or []
            organic_norm = []
            for r in pages:
                organic_norm.append(
                    {
                        "title": r.get("name", ""),
                        "link": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
            return True, organic_norm, None

        raw = self._run_with_429_retry("Bing", fetch_once)
        if isinstance(raw, dict) and "error" in raw:
            return raw
        return _normalize_organic_results(self.show_snippet, raw or [], max_results)

    def _search_brave(
        self, keywords: str, max_results: Optional[int], region: Optional[str]
    ):
        key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not key:
            return {"error": "BRAVE_SEARCH_API_KEY is not set."}

        count = max(1, min(max_results or 10, 20))
        params: dict[str, Any] = {"q": keywords, "count": count}
        gl, hl = _region_to_gl_hl(region)
        if gl:
            params["country"] = gl.upper()
        if hl:
            params["search_lang"] = hl

        def fetch_once():
            try:
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": key,
                    },
                    params=params,
                    timeout=60,
                )
                if resp.status_code == 429:
                    return False, None, resp.text
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return False, None, str(e)

            web = data.get("web") or {}
            rows = web.get("results") or []
            organic_norm = []
            for r in rows:
                organic_norm.append(
                    {
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "snippet": r.get("description", ""),
                    }
                )
            return True, organic_norm, None

        raw = self._run_with_429_retry("Brave", fetch_once)
        if isinstance(raw, dict) and "error" in raw:
            return raw
        return _normalize_organic_results(self.show_snippet, raw or [], max_results)

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Backends (WEB_SEARCH_BACKEND env): serpapi (default), serper, tavily, bing, brave.
        All backends normalize to the same result list shape as legacy SerpAPI organic_results.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        backend = os.getenv(WEB_SEARCH_BACKEND_ENV, "serpapi").strip().lower()
        if backend in ("", "serpapi"):
            return self._search_serpapi(keywords, max_results, region)
        if backend == "serper":
            return self._search_serper(keywords, max_results, region)
        if backend == "tavily":
            return self._search_tavily(keywords, max_results, region)
        if backend == "bing":
            return self._search_bing(keywords, max_results, region)
        if backend == "brave":
            return self._search_brave(keywords, max_results, region)
        return {
            "error": f"Unsupported WEB_SEARCH_BACKEND={backend!r}. "
            f"Use serpapi, serper, tavily, bing, or brave."
        }

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            # A header that mimics a browser request. This helps avoid 403 Forbidden errors.
            # TODO: Is this the best way to do this?
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()

            # Note: Un-comment this when we want to simulate a random error
            # Flip a coin to simulate a random error
            # if self._random.random() < 0.95:
            #     return {"error": self._fake_requests_get_error_msg(url)}

            # Process the response based on the mode
            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic‑looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)
