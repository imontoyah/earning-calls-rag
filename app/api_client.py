"""Thin HTTP client for the Earnings Call RAG FastAPI backend.

Keeps the Streamlit layer free of transport concerns: URL building, auth
headers, timeouts, and turning HTTP errors into friendly, user-facing
messages all live here.
"""

from __future__ import annotations

import requests

DEFAULT_TIMEOUT = 90  # LLM calls can be slow; keep generous.


class APIError(Exception):
    """Raised for any non-2xx response or transport failure.

    Carries the HTTP status code (when there is one) so the UI can react
    differently to auth vs. rate-limit vs. validation errors.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _friendly_error(resp: requests.Response) -> APIError:
    """Translate a FastAPI/slowapi error response into a helpful message."""
    detail = None
    try:
        body = resp.json()
        detail = body.get("detail") or body.get("error")
    except ValueError:
        detail = resp.text or None

    messages = {
        401: "API key faltante. Ingresa la clave en la barra lateral.",
        403: "API key inválida. Revisa la clave en la barra lateral.",
        409: detail or "El recurso ya existe.",
        422: f"Datos inválidos: {detail}" if detail else "Datos inválidos.",
        429: "Límite de peticiones alcanzado. Espera un momento e intenta de nuevo.",
        503: "El servidor no tiene API_KEY configurada. Es un problema del backend.",
    }
    message = messages.get(resp.status_code, detail or f"Error {resp.status_code}.")
    return APIError(message, status_code=resp.status_code)


class RAGClient:
    """Typed wrapper around the RAG API endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout

    # -- internals ---------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError:
            raise APIError(
                f"No pude conectar con la API en {self.base_url}. "
                "¿Está corriendo? Inicia con: uv run fastapi dev api/main.py"
            ) from None
        except requests.exceptions.Timeout:
            raise APIError("La API tardó demasiado en responder (timeout).") from None

        if resp.status_code >= 400:
            raise _friendly_error(resp)
        return resp.json()

    # -- public endpoints --------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def collections(self) -> dict:
        return self._request("GET", "/collections")

    def ask(
        self,
        question: str,
        company: str | None = None,
        quarter: str | None = None,
        n_results: int = 5,
    ) -> dict:
        payload: dict = {"question": question, "n_results": n_results}
        if company:
            payload["company"] = company
        if quarter:
            payload["quarter"] = quarter
        return self._request("POST", "/ask", json=payload, headers=self._auth_headers())

    def ask_temporal(
        self,
        question: str,
        quarters: list[str],
        company: str | None = None,
        n_per_quarter: int = 2,
    ) -> dict:
        payload: dict = {
            "question": question,
            "quarters": quarters,
            "n_per_quarter": n_per_quarter,
        }
        if company:
            payload["company"] = company
        return self._request(
            "POST", "/ask/temporal", json=payload, headers=self._auth_headers()
        )

    def ingest(self, url: str, ticker: str, quarter: str, date: str) -> dict:
        payload = {"url": url, "ticker": ticker, "quarter": quarter, "date": date}
        return self._request("POST", "/ingest", json=payload, headers=self._auth_headers())
