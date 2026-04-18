import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config.settings import FINNHUB_API_KEY, SEARCH_DEFAULT_QUERY, SEARCH_RESULT_LIMIT


FINNHUB_SEARCH_URL = "https://finnhub.io/api/v1/search"


class SymbolSearchError(RuntimeError):
    pass


def _clean_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def search_symbols(query: str) -> list[dict[str, str]]:
    cleaned_query = query.strip() or SEARCH_DEFAULT_QUERY

    if not FINNHUB_API_KEY:
        raise SymbolSearchError("Finnhub API key is missing. Set FINNHUB_API_KEY in your environment.")

    params = urlencode({"q": cleaned_query, "token": FINNHUB_API_KEY})
    request = Request(f"{FINNHUB_SEARCH_URL}?{params}", headers={"User-Agent": "stock-dashboard/1.0"})

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise SymbolSearchError("Invalid Finnhub API key. Check FINNHUB_API_KEY.") from exc
        if exc.code == 429:
            raise SymbolSearchError("Finnhub rate limit reached. Try again shortly.") from exc
        raise SymbolSearchError("Unable to fetch search results from Finnhub.") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SymbolSearchError("Unable to fetch search results from Finnhub.") from exc

    results = payload.get("result", [])
    matches = []
    seen_symbols = set()
    for item in results:
        symbol = _clean_symbol(str(item.get("symbol", "")))
        description = str(item.get("description", "")).strip()
        if not symbol or not description or symbol in seen_symbols:
            continue
        matches.append({"symbol": symbol, "description": description})
        seen_symbols.add(symbol)
        if len(matches) >= SEARCH_RESULT_LIMIT:
            break

    return matches
