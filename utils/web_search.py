from duckduckgo_search import DDGS


def search_web(query):
    """Search the web using DuckDuckGo"""
    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=3)

            for result in search_results:
                results.append(result["body"])

        return "\n".join(results)

    except Exception as e:
        return f"Web search error: {str(e)}"