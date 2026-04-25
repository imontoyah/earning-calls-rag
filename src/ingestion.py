"""Download and parse earnings call transcripts from Motley Fool."""

import re
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path


HEADERS = {"User-Agent": "Mozilla/5.0 (educational-project)"}


def download_transcript(url: str) -> str:
    """Download HTML from a Motley Fool transcript URL."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text


def extract_article_text(html: str) -> str:
    """Extract the article body text from Motley Fool HTML."""
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


def _extract_participant_map(full_text: str) -> dict[str, str]:
    """Extract a name -> role mapping from the Call Participants section.

    New-format transcripts list participants as:
        Chief Executive Officer — Timothy D. Cook
        Chief Financial Officer — Kevan Parekh
    """
    participants = {}
    for match in re.finditer(r"(.+?)\s—\s(.+)", full_text):
        role = match.group(1).strip()
        name = match.group(2).strip()
        participants[name] = role
    return participants


def parse_speaker_turns(text: str, participant_map: dict[str, str]) -> list[dict]:
    """Parse speaker turns from new format: Name: text...

    Splits on lines where a capitalized name is followed by a colon.
    Looks up role from participant_map; empty string if not found.
    """
    turns = []

    # Build regex from known participant names + generic name pattern for unknown speakers
    known_names = [re.escape(name) for name in participant_map]
    # Generic pattern: 1-5 capitalized words (with optional dots for initials like "D.")
    generic_pattern = r"(?:[A-Z][a-zA-Z.]+\s?){1,5}"

    if known_names:
        # Match known names OR generic capitalized names
        name_pattern = "|".join(known_names) + "|" + generic_pattern
    else:
        name_pattern = generic_pattern

    pattern = r"\n(" + name_pattern + r"):\n"

    parts = re.split(pattern, text)

    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        spoken_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        role = participant_map.get(name, "")
        turn = {
            "speaker": name,
            "role": role,
            "text": spoken_text,
        }
        if len(turn["text"]) > 10:
            turns.append(turn)

    return turns


def process_transcript(url: str, company: str, quarter: str, date: str) -> dict:
    """Full pipeline: download, parse, and structure a single transcript."""
    html = download_transcript(url)
    full_text = extract_article_text(html)

    participant_map = _extract_participant_map(full_text)
    turns = parse_speaker_turns(full_text, participant_map)

    return {
        "company": company,
        "quarter": quarter,
        "date": date,
        "source_url": url,
        "total_turns": len(turns),
        "speakers": list(set(t["speaker"] for t in turns)),
        "turns": turns,
    }


def save_transcript(transcript: dict, output_dir: str = "data/processed") -> Path:
    """Save a transcript dict as JSON."""
    output_path = Path(output_dir) / f"{transcript['company']}_{transcript['quarter'].replace('-', '_')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    return output_path
