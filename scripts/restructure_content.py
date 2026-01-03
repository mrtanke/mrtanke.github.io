from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"

TARGETS = ["posts", "projects"]


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    text = ascii_text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text or "untitled"


def extract_metadata(md_path: Path) -> tuple[str, str]:
    text = md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        raise ValueError(f"{md_path} is missing front matter")
    data = yaml.safe_load(match.group(1)) or {}
    title = data.get("title")
    date_val = data.get("date")
    if not title or not date_val:
        raise ValueError(f"{md_path} missing title/date in front matter")

    if hasattr(date_val, "strftime"):
        date_part = date_val.strftime("%Y-%m-%d")
    else:
        date_str = str(date_val)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_str)
        if not date_match:
            raise ValueError(f"{md_path} has unparsable date: {date_str}")
        date_part = date_match.group(0)

    slug = slugify(title)
    return date_part, slug


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}-{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def convert_file(md_file: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=False)
    shutil.move(str(md_file), target_dir / "index.md")


def rename_tree():
    for section in TARGETS:
        base_dir = CONTENT / section
        if not base_dir.exists():
            continue
        entries = sorted(base_dir.iterdir())
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                md_path = entry / "index.md"
                if not md_path.exists():
                    continue
            else:
                if entry.suffix.lower() != ".md":
                    continue
                md_path = entry

            date_part, slug = extract_metadata(md_path)
            new_name = f"{date_part}-{slug}"
            target_path = base_dir / new_name
            if entry.is_dir() and entry.resolve() == target_path.resolve():
                continue
            if target_path.exists():
                target_path = ensure_unique(target_path)

            if entry.is_dir():
                entry.rename(target_path)
            else:
                convert_file(entry, target_path)


if __name__ == "__main__":
    rename_tree()
