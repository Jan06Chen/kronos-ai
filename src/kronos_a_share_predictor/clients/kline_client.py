from __future__ import annotations

import requests


class KlineClient:
    def __init__(self, api_base_url: str, timeout: int) -> None:
        self._api_base_url = api_base_url
        self._timeout = timeout

    def fetch_kline(self, stock_code: str, starttime: str, endtime: str) -> list[dict]:
        response = requests.get(
            f"{self._api_base_url}/stock/kline",
            params={
                "stock_code": stock_code,
                "start_date": starttime,
                "end_date": endtime,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 20000 and payload.get("success") is not True:
            raise ValueError(f"股票 {stock_code} 的 K 线接口返回失败: {payload}")

        items = payload.get("data", {}).get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"股票 {stock_code} 的 K 线响应中 data.items 不是数组")

        return items
