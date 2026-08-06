from typing import Any

import httpx


class HhApiClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        user_agent: str,
        access_token: str | None = None,
    ) -> None:
        self._client = http_client
        self._headers = {
            "HH-User-Agent": user_agent,
        }

        if access_token:
            self._headers["Authorization"] = f"Bearer {access_token}"

    async def search_vacancies(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = await self._client.get(
            "https://api.hh.ru/vacancies",
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_vacancy(
        self,
        vacancy_id: str,
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"https://api.hh.ru/vacancies/{vacancy_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()
