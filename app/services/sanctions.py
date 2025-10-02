from typing import List, Optional

import requests

from app.schemas import SanctionsMatch


def query_opensanctions(name: str, birth_date: Optional[str], api_key: Optional[str], top_k: int = 5) -> List[SanctionsMatch]:
    """
    Call OpenSanctions match API. Falls back gracefully on errors.
    Docs: https://www.opensanctions.org/docs/api/
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    payload = {
        "query": {
            "name": name,
        },
        "size": top_k
    }
    if birth_date:
        payload["query"]["birthDate"] = birth_date

    # POST /match
    url = "https://api.opensanctions.org/match"
    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for res in data.get("results", []):
        # unified structure
        match = SanctionsMatch(
            id=res.get("id"),
            name=res.get("name") or (res.get("entity", {}) or {}).get("name"),
            country=(res.get("entity", {}) or {}).get("country"),
            dataset=res.get("dataset"),
            schema=(res.get("entity", {}) or {}).get("schema"),
            score=res.get("score"),
            link=res.get("target", {}).get("url") if res.get("target") else None,
        )
        results.append(match)

    # Sort by score desc if present
    results.sort(key=lambda m: (m.score or 0.0), reverse=True)
    return results