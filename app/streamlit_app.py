"""Streamlit UI for the Earnings Call RAG service (Phase 6.1).

A thin frontend that consumes the FastAPI backend over HTTP, so it inherits
the auth, rate-limiting, and SSRF protections built in Phase 5. Configure the
backend with the API_BASE_URL / API_KEY env vars, or from the sidebar.
"""

from __future__ import annotations

import os
import sys
from datetime import date as date_cls
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts app/ on sys.path but not the
# project root, so make the root importable regardless of the launch CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.api_client import APIError, RAGClient

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Earnings Call RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Answer card */
      .answer-card {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.20);
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.5rem;
      }
      /* Connection pill */
      .pill { display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:0.78rem; font-weight:600; }
      .pill-ok   { background:#dcfce7; color:#166534; }
      .pill-down { background:#fee2e2; color:#991b1b; }
      .muted { color:#64748b; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_client() -> RAGClient:
    return RAGClient(
        base_url=st.session_state.get("base_url", DEFAULT_BASE_URL),
        api_key=st.session_state.get("api_key", DEFAULT_API_KEY),
    )


@st.cache_data(ttl=30, show_spinner=False)
def fetch_collections(base_url: str) -> dict:
    """Company/quarter catalog. Cached briefly; cleared after an ingest."""
    return RAGClient(base_url).collections()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(base_url: str) -> dict:
    return RAGClient(base_url).health()


def companies_map(base_url: str) -> dict[str, list[str]]:
    """{ticker: [quarters]} from /collections, or {} if unreachable."""
    try:
        data = fetch_collections(base_url)
    except APIError:
        return {}
    return {c["ticker"]: c["quarters"] for c in data.get("companies", [])}


# ---------------------------------------------------------------------------
# Sidebar — connection control panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Earnings Call RAG")
    st.caption("Interfaz para consultar transcripciones de earnings calls.")

    st.subheader("Conexión")
    st.text_input("API base URL", value=DEFAULT_BASE_URL, key="base_url")
    st.text_input(
        "API key",
        value=DEFAULT_API_KEY,
        key="api_key",
        type="password",
        help="Requerida para preguntar e ingestar. Header X-API-Key.",
    )

    base_url = st.session_state["base_url"]
    try:
        health = fetch_health(base_url)
        st.markdown(
            f'<span class="pill pill-ok">● Conectado</span> '
            f'<span class="muted">{health["collection_count"]} chunks indexados</span>',
            unsafe_allow_html=True,
        )
    except APIError as e:
        st.markdown('<span class="pill pill-down">● Sin conexión</span>', unsafe_allow_html=True)
        st.caption(str(e))

    if st.button("↻ Actualizar datos", use_container_width=True):
        fetch_collections.clear()
        fetch_health.clear()
        st.rerun()


companies = companies_map(base_url)
tickers = sorted(companies.keys())


def render_answer(answer: str) -> None:
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_ask, tab_temporal, tab_ingest, tab_status = st.tabs(
    ["💬 Preguntar", "📈 Temporal", "➕ Ingestar", "📊 Estado"]
)

# --- Tab: Ask --------------------------------------------------------------
with tab_ask:
    st.subheader("Pregunta sobre una earnings call")
    st.caption("Recuperación semántica sobre los transcripts indexados.")

    question = st.text_area(
        "Pregunta",
        placeholder="¿Qué dijo el CFO sobre los márgenes este trimestre?",
        key="ask_question",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        company = st.selectbox("Empresa", ["Todas", *tickers], key="ask_company")
    with c2:
        q_options = companies.get(company, []) if company != "Todas" else []
        quarter = st.selectbox("Quarter", ["Todos", *q_options], key="ask_quarter")
    with c3:
        n_results = st.slider("Fragmentos (n_results)", 1, 20, 5, key="ask_n")

    if st.button("Preguntar", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Escribe una pregunta primero.")
        else:
            try:
                with st.spinner("Consultando el modelo…"):
                    res = get_client().ask(
                        question=question,
                        company=None if company == "Todas" else company,
                        quarter=None if quarter == "Todos" else quarter,
                        n_results=n_results,
                    )
                render_answer(res["answer"])
            except APIError as e:
                st.error(str(e))

# --- Tab: Temporal ---------------------------------------------------------
with tab_temporal:
    st.subheader("Comparar entre trimestres")
    st.caption("Recupera contexto por quarter para identificar tendencias.")

    t_company = st.selectbox("Empresa", ["Todas", *tickers], key="temp_company")
    available_quarters = (
        companies.get(t_company, [])
        if t_company != "Todas"
        else sorted({q for qs in companies.values() for q in qs})
    )
    t_quarters = st.multiselect(
        "Trimestres a comparar",
        available_quarters,
        key="temp_quarters",
        help="Elige 2 o más para ver la evolución.",
    )
    t_question = st.text_area(
        "Pregunta",
        placeholder="¿Cómo evolucionó el guidance de ingresos entre trimestres?",
        key="temp_question",
    )
    n_per_quarter = st.slider("Fragmentos por quarter", 1, 10, 2, key="temp_n")

    if st.button("Comparar", type="primary", key="temp_btn"):
        if not t_question.strip():
            st.warning("Escribe una pregunta primero.")
        elif not t_quarters:
            st.warning("Selecciona al menos un trimestre.")
        else:
            try:
                with st.spinner("Analizando trimestres…"):
                    res = get_client().ask_temporal(
                        question=t_question,
                        quarters=t_quarters,
                        company=None if t_company == "Todas" else t_company,
                        n_per_quarter=n_per_quarter,
                    )
                render_answer(res["answer"])
            except APIError as e:
                st.error(str(e))

# --- Tab: Ingest -----------------------------------------------------------
with tab_ingest:
    st.subheader("Ingestar una nueva transcripción")
    st.caption("Solo fuentes permitidas (fool.com). Requiere API key de escritura.")

    with st.form("ingest_form"):
        url = st.text_input(
            "URL del transcript",
            placeholder="https://www.fool.com/earnings/call-transcripts/…",
        )
        f1, f2 = st.columns(2)
        with f1:
            ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        with f2:
            call_date = st.date_input("Fecha de la call", value=date_cls.today())
        f3, f4 = st.columns(2)
        with f3:
            q_num = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
        with f4:
            year = st.number_input("Año", min_value=2000, max_value=2100,
                                   value=date_cls.today().year, step=1)
        submitted = st.form_submit_button("Ingestar", type="primary")

    if submitted:
        if not url.strip() or not ticker:
            st.warning("URL y ticker son obligatorios.")
        else:
            quarter = f"{q_num}-{year}"
            try:
                with st.spinner("Descargando, parseando e indexando…"):
                    res = get_client().ingest(
                        url=url.strip(),
                        ticker=ticker,
                        quarter=quarter,
                        date=call_date.isoformat(),
                    )
                st.success(f"Ingestado {res['ticker']} {res['quarter']} ✅")
                m1, m2 = st.columns(2)
                m1.metric("Turnos", res["total_turns"])
                m2.metric("Chunks indexados", res["chunks_indexed"])
                fetch_collections.clear()
                fetch_health.clear()
            except APIError as e:
                st.error(str(e))

# --- Tab: Status -----------------------------------------------------------
with tab_status:
    st.subheader("Estado del servicio")
    try:
        health = fetch_health(base_url)
        data = fetch_collections(base_url)
        c1, c2 = st.columns(2)
        c1.metric("Estado", health["status"].upper())
        c2.metric("Chunks indexados", health["collection_count"])

        st.markdown("**Empresas disponibles**")
        rows = [
            {"Ticker": c["ticker"], "Trimestres": ", ".join(c["quarters"]),
             "Nº quarters": len(c["quarters"])}
            for c in data.get("companies", [])
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No hay transcripciones indexadas todavía. Usa la pestaña Ingestar.")
    except APIError as e:
        st.error(str(e))
