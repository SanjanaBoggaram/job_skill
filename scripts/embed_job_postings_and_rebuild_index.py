from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")
    return api_key


def embed_role(openrouter_api_key: str, role_text: str, embedding_model: str) -> list[float]:
    """Call OpenRouter embeddings endpoint via raw HTTP.

    The OpenAI Python SDK sometimes fails to parse OpenRouter embedding payloads (data=None).
    This implementation extracts the vector directly from the JSON response.
    """
    url = "https://openrouter.ai/api/v1/embeddings"
    payload = {"model": embedding_model, "input": role_text}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {openrouter_api_key}")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)

    # Expected: {"data": [{"embedding": [...], ...}], ...}
    vec = None
    if isinstance(parsed, dict):
        data_arr = parsed.get("data")
        if isinstance(data_arr, list) and data_arr:
            first = data_arr[0]
            if isinstance(first, dict):
                vec = first.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"Unexpected embeddings response shape: {parsed.keys() if isinstance(parsed, dict) else type(parsed)}")

    return [float(x) for x in vec]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adds role_embedding to each JSON in outputs/job_postings and rebuilds outputs/job_postings_index.json"
        )
    )
    parser.add_argument(
        "--postings-dir",
        default="outputs/job_postings",
        help="Directory containing per-job JSON files",
    )
    parser.add_argument(
        "--index-out",
        default="outputs/job_postings_index.json",
        help="Output index JSON path",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("JOB_ROLE_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free"),
        help="OpenRouter embedding model id",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Optional sleep between API calls (seconds)",
    )

    args = parser.parse_args()

    load_dotenv()

    postings_dir = Path(args.postings_dir)
    if not postings_dir.exists():
        raise FileNotFoundError(f"postings-dir not found: {postings_dir}")

    embedding_model = str(args.embedding_model).strip()
    if not embedding_model:
        raise ValueError("embedding-model cannot be empty")

    openrouter_api_key = get_openrouter_api_key()

    job_files = sorted([p for p in postings_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    if not job_files:
        raise RuntimeError(f"No job JSON files found in {postings_dir}")

    index_items: list[dict[str, Any]] = []

    for i, path in enumerate(job_files, start=1):
        job = read_json(path)
        if not isinstance(job, dict):
            print(f"Skipping non-object JSON: {path.name}")
            continue

        role = str(job.get("role", "")).strip()
        job_id = str(job.get("job_id", path.stem)).strip() or path.stem

        if not role:
            # Fallback to job_id for embedding.
            role = job_id

        # Compute embedding (retry once).
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                vec = embed_role(openrouter_api_key, role_text=role, embedding_model=embedding_model)
                job["role_embedding"] = vec
                job["embedding_model"] = embedding_model
                break
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(1.0)
                else:
                    raise

        write_json(path, job)

        # Build index entry from the updated job JSON.
        item = dict(job)
        item.setdefault("job_id", job_id)
        item["json_path"] = str(path.as_posix())
        index_items.append(item)

        print(f"[{i}/{len(job_files)}] embedded role for {job_id}")
        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    index_out = Path(args.index_out)
    write_json(index_out, {"items": index_items})
    print(f"Wrote index: {index_out} (items={len(index_items)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
