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
    """Split transcript text into speaker turns."""
    turns = []
    parts = re.split(r'([A-Z][a-zA-Z .]+\n--\n[A-Za-z ,]+)', text)
    for i in range(1, len(parts), 2):
        name, role = parts[i].split("--", 1)
        turn = {
            "speaker": name.strip(),
            "role": role.strip(),
            "text": parts[i + 1].strip() if i + 1 < len(parts) else "",
        }
        if len(turn["text"]) > 10:
            turns.append(turn)
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


def save_transcript(transcript: dict, output_dir: Path = DATA_DIR) -> Path:
    """Persist a transcript dict as JSON under data/processed/."""
    path = output_dir / f"{transcript['company']}_{transcript['quarter'].replace('-', '_')}.json"
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