from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedSkill:
    skill: str
    parents: list[str]
    score: int


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from model output."""
    cleaned = strip_code_fences(text)
    cleaned = cleaned.strip()
    # Try direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find the outermost {...} region.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse a JSON object from model output")


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks.append(page_text)
    return "\n".join(chunks)


def read_posting_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def load_ontology(ontology_path: Path) -> dict[str, list[str]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology: dict[str, list[str]] = {}
    for skill_name, meta in data.items():
        parents = meta.get("parents", [])
        if not isinstance(parents, list):
            parents = []
        ontology[str(skill_name)] = [str(p) for p in parents]
    return ontology


def _case_insensitive_ontology_lookup(ontology: dict[str, list[str]], skill: str) -> list[str] | None:
    direct = ontology.get(skill)
    if direct is not None:
        return direct
    s = skill.casefold()
    for k, parents in ontology.items():
        if k.casefold() == s:
            return parents
    return None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Keep in sync with scripts/extract_resume_skills.py
OPENROUTER_CHAT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
OPENROUTER_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


def llm_extract_job_posting(
    posting_text: str,
    ontology: dict[str, list[str]],
    client: Any,
) -> dict[str, Any]:
    ontology_json = json.dumps(ontology, ensure_ascii=False)

    system = (
        "Extract structured information from job postings. "
        "Return ONLY valid JSON (no markdown) with keys: company_name, role, location, employment_type, skills. "
        "skills must be an array of objects: {skill: string, parents: string[], score: int 0-10}. "
        "Use this ontology (JSON mapping skill->parents) to choose parents whenever applicable:\n"
        f"{ontology_json}\n\n"
        "If a skill is not in the ontology, choose the closest parent categories from the ontology, or [] if none. "
        "Score guidance: 9-10 required/central; 7-8 strongly preferred; 5-6 mentioned once/as plus; 1-4 tangential."
    )

    human = f"Job posting text:\n\n{posting_text}\n\nReturn only JSON."

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            print(f"  - LLM extract (attempt {attempt + 1}/3)...", flush=True)
            response = client.chat.completions.create(
                model=OPENROUTER_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. All keys and strings MUST use double quotes. No text outside JSON."},
                    {"role": "user", "content": f"{system}\n\n{human}"},
                ],
                temperature=0.0,
                max_tokens=1800,
            )
            raw = strip_code_fences(response.choices[0].message.content or "")
            print("\n===== LLM RAW OUTPUT START =====")
            print(raw)
            print("===== LLM RAW OUTPUT END =====\n")
            if not raw.strip():
                raise ValueError("LLM returned empty response")
            return extract_json_object(raw)
        except Exception as e:  # pragma: no cover
            last_err = e
            human = "IMPORTANT: Output must be ONLY valid JSON. No markdown, no extra text.\n\n" + human
            time.sleep(0.75 * (attempt + 1))
    assert last_err is not None
    raise last_err


def embed_text(openrouter_api_key: str, text: str) -> list[float]:
    import requests

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            print(f"  - Role embedding (attempt {attempt + 1}/3)...", flush=True)
            r = requests.post(
                f"{OPENROUTER_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": OPENROUTER_EMBED_MODEL, "input": text},
                timeout=60,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"OpenRouter embeddings error {r.status_code}: {r.text}")

            data = r.json()
            items = data.get("data")
            if not isinstance(items, list) or not items:
                raise RuntimeError(f"Unexpected embeddings response: {data}")
            emb = items[0].get("embedding")
            if not isinstance(emb, list) or not emb:
                raise RuntimeError(f"Missing embedding in response: {data}")
            return [float(x) for x in emb]
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(0.75 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _coerce_skills(skills_raw: Any, ontology: dict[str, list[str]]) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    if not isinstance(skills_raw, list):
        return skills

    for s in skills_raw:
        if not isinstance(s, dict):
            continue
        name = str(s.get("skill", "")).strip()
        if not name:
            continue

        parents_val = s.get("parents", [])
        if isinstance(parents_val, str):
            parents = [parents_val]
        elif isinstance(parents_val, list):
            parents = [str(p).strip() for p in parents_val if str(p).strip()]
        else:
            parents = []

        # Overwrite parents with ontology if it exists (keeps taxonomy consistent).
        onto_parents = _case_insensitive_ontology_lookup(ontology, name)
        if onto_parents is not None:
            parents = onto_parents

        try:
            score = int(s.get("score", 0))
        except Exception:
            score = 0
        score = max(0, min(10, score))

        skills.append({"skill": name, "parents": parents, "score": score})

    # De-dup by skill name (case-insensitive), keep the highest score.
    best: dict[str, dict[str, Any]] = {}
    for item in skills:
        key = str(item["skill"]).casefold()
        prev = best.get(key)
        if prev is None or int(item.get("score", 0)) > int(prev.get("score", 0)):
            best[key] = item

    return sorted(best.values(), key=lambda x: (-int(x.get("score", 0)), str(x.get("skill", "")).casefold()))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build outputs/job_postings_index.json with skills + role embeddings.")
    parser.add_argument("--in-dir", default="data/job_postings", help="Directory containing job posting files")
    parser.add_argument("--ontology", default="data/skill_ontology.json", help="Skill ontology JSON")
    parser.add_argument("--out", default="outputs/job_postings_index.json", help="Output index JSON path")

    args = parser.parse_args()

    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    from openai import OpenAI

    client = OpenAI(
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=120,
    )

    in_dir = Path(args.in_dir)
    ontology = load_ontology(Path(args.ontology))

    posting_paths = sorted([p for p in in_dir.iterdir() if p.is_file()])
    items: list[dict[str, Any]] = []
    for idx, path in enumerate(posting_paths, start=1):
        job_id = path.name
        print(f"[{idx}/{len(posting_paths)}] Processing {job_id}...")
        posting_text = _normalize_whitespace(read_posting_text(path))

        extracted = llm_extract_job_posting(posting_text, ontology, client=client)
        company_name = str(extracted.get("company_name", "")).strip()
        role = str(extracted.get("role", "")).strip()
        location = str(extracted.get("location", "")).strip()
        employment_type = str(extracted.get("employment_type", "")).strip()
        skills = _coerce_skills(extracted.get("skills", []), ontology)

        role_embedding = embed_text(openrouter_key, role or job_id)
        print(f"  - Done {job_id}", flush=True)

        items.append(
            {
                "job_id": job_id,
                "company_name": company_name,
                "role": role,
                "location": location,
                "employment_type": employment_type,
                "skills": skills,
                "role_embedding": role_embedding,
                "embedding_model": OPENROUTER_EMBED_MODEL,
                "source_path": str(path.as_posix()),
            }
        )

    out_path = Path(args.out)
    write_json(out_path, {"items": items})
    print(f"Built index for {len(items)} job postings")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
