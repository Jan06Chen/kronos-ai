from __future__ import annotations

from collections import OrderedDict

import requests


class RecommendationClient:
    def __init__(self, api_base_url: str, timeout: int) -> None:
        self._api_base_url = api_base_url
        self._timeout = timeout

    def _get_payload(self, path: str, params: dict | None = None) -> dict:
        response = requests.get(
            f"{self._api_base_url}{path}",
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 20000 and payload.get("success") is not True:
            raise ValueError(f"接口返回失败: path={path} payload={payload}")
        return payload

    @staticmethod
    def _extract_items(payload: dict) -> list[dict]:
        data = payload.get("data")
        if isinstance(data, dict):
            items = data.get("items", data.get("list", data.get("rows", data)))
        else:
            items = data

        if isinstance(items, list):
            return items
        if isinstance(items, dict):
            nested_items = items.get("items")
            if isinstance(nested_items, list):
                return nested_items
        raise ValueError("响应中无法解析股票列表，data/items 不是数组")

    @staticmethod
    def _dedupe_stock_codes(items: list[dict]) -> list[str]:
        return list(
            OrderedDict.fromkeys(
                item.get("stock_code") or item.get("code") or item.get("symbol") or item.get("ts_code")
                for item in items
                if isinstance(item, dict)
                and (item.get("stock_code") or item.get("code") or item.get("symbol") or item.get("ts_code"))
            )
        )

    def fetch_stock_codes(self, recommendation_date: str | None = None) -> tuple[list[str], list[dict]]:
        if recommendation_date:
            payload = self._get_payload("/recommendations", params={"date": recommendation_date})
            items = self._extract_items(payload)
            return self._dedupe_stock_codes(items), items

        payload = self._get_payload("/stock/list")
        items = self._extract_items(payload)
        return self._dedupe_stock_codes(items), items
