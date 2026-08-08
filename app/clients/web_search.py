"""DuckDuckGo-based web search — free, no API key."""

from typing import Any

import httpx


SEARCH_TIMEOUT = 8


def search_web(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Run a DuckDuckGo text search and return snippet summaries.

    Returns an empty list on any failure — callers should degrade gracefully."""
    try:
        response = httpx.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=SEARCH_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    results: list[dict[str, str]] = []

    # Abstract + source
    abstract = str(data.get("AbstractText") or data.get("Abstract") or "").strip()
    source = str(data.get("AbstractURL") or data.get("AbstractSource") or "").strip()
    if abstract:
        results.append({"snippet": abstract, "source": source})

    # Related topics
    for topic in (data.get("RelatedTopics") or [])[:max_results]:
        if isinstance(topic, dict):
            text = str(topic.get("Text") or "").strip()
            url = str(topic.get("FirstURL") or "").strip()
            if text:
                results.append({"snippet": text, "source": url})

    return results[:max_results]
