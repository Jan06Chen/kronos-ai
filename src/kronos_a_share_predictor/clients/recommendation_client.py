from __future__ import annotations

from collections import OrderedDict

import requests


class RecommendationClient:
    def __init__(self, api_base_url: str, timeout: int) -> None:
        self._api_base_url = api_base_url
        self._timeout = timeout

    def fetch_stock_codes(self, recommendation_date: str) -> tuple[list[str], list[dict]]:
        response = requests.get(
            f"{self._api_base_url}/recommendations",
            params={"date": recommendation_date},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 20000 and payload.get("success") is not True:
            raise ValueError(f"recommendations 接口返回失败: {payload}")

        items = payload.get("data", {}).get("items", [])
        if not isinstance(items, list):
            raise ValueError("recommendations 响应中 data.items 不是数组")

        deduped_codes = list(OrderedDict.fromkeys(item["stock_code"] for item in items if item.get("stock_code")))
        return deduped_codes, items
