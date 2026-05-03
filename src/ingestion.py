"""Download, parse, and save earnings call transcripts."""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import COMPANIES_FILE, DATA_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (educational-project)"}


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def download_transcript(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = (
        soup.find("div", class_="article-body")
        or soup.find("article")
        or soup.find("div", class_="tailwind-article-body")
    )
    if article is None:
        for div in soup.find_all("div"):
            if div.find("strong", string=re.compile(r".+\s--\s.+")):
                article = div
                break
    if article is None:
        raise ValueError("Could not find article body in HTML")
    return article.get_text(separator="\n", strip=True)


def parse_speaker_turns(text: str) -> list[dict]:
    """Split transcript text into speaker turns.

    Handles the Motley Fool format where the participant list uses
    'Role — Name' headers and each turn in the body is rendered by
    get_text() as 'Name:\\n<speech>'.
    """
    role_map = {}
    for m in re.finditer(r'^(.+?)\s*—\s*([A-Z][a-zA-Z .\-]+)\s*$', text, re.MULTILINE):
        role_map[m.group(2).strip()] = m.group(1).strip()

    marker = "Full Conference Call Transcript"
    body = text[text.find(marker) + len(marker):] if marker in text else text

    parts = re.split(r'\n([A-Z][a-zA-Z .\-]+):\n', body)

    turns = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        speech = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if len(speech) > 10:
            turns.append({
                "speaker": name,
                "role": role_map.get(name, ""),
                "text": speech,
            })
    return turns


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_transcript(url: str, company: str, quarter: str, date: str) -> dict:
    """Full pipeline: download → parse → return structured dict."""
    html = download_transcript(url)
    text = extract_article_text(html)
    turns = parse_speaker_turns(text)
    return {
        "company": company,
        "quarter": quarter,
        "date": date,
        "source_url": url,
        "total_turns": len(turns),
        "speakers": list(set(t["speaker"] for t in turns)),
        "turns": turns,
    }


def save_transcript(transcript: dict, output_dir: Path | str = DATA_DIR) -> Path:
    """Persist a transcript dict as JSON under data/processed/."""
    path = Path(output_dir) / f"{transcript['company']}_{transcript['quarter'].replace('-', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Multi-company config
# ---------------------------------------------------------------------------

def load_companies(config_path: Path = COMPANIES_FILE) -> list[dict]:
    """
    Read companies.json and return a flat list of transcript entries.

    Each entry is a dict with keys: ticker, url, quarter, date.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    flat_dict = []
    for company in data["companies"]:
        for trans in company["transcripts"]:
            single_dict = {"ticker": company["ticker"], 
                        "url": trans["url"],
                        "quarter": trans["quarter"], 
                        "date": trans["date"]}
            flat_dict.append(single_dict)
    

    return flat_dict