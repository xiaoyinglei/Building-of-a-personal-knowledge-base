from __future__ import annotations

import httpx


class WebFetchRepo:
    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._http_client = http_client
        self._timeout = timeout

    def fetch(self, location: str) -> str:
        response = self._client().get(location, follow_redirects=True, timeout=self._timeout)
        response.raise_for_status()
        return response.text

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client()
        return self._http_client
