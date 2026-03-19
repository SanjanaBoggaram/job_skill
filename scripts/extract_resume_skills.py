from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedSkill:
    skill: str
    parents: list[str]
    score: int


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
    return "\n".join(chunks)


def load_ontology(ontology_path: Path) -> dict[str, list[str]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology: dict[str, list[str]] = {}
    for skill_name, meta in data.items():
        parents = meta.get("parents", [])
        if not isinstance(parents, list):
            parents = []
        ontology[str(skill_name)] = [str(p) for p in parents]
    return ontology


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    t = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _case_insensitive_ontology_lookup(ontology: dict[str, list[str]], skill: str) -> list[str] | None:
    direct = ontology.get(skill)
    if direct is not None:
        return direct
    lowered = skill.casefold()
    for k, parents in ontology.items():
        if k.casefold() == lowered:
            return parents
    return None


def heuristic_extract_skills(resume_text: str, ontology: dict[str, list[str]]) -> list[ExtractedSkill]:
    text = resume_text

    # Seed with ontology skill names that appear in the resume.
    found: dict[str, int] = {}
    for skill in ontology.keys():
        # Word boundary-ish match, but allow spaces and symbols inside the skill string.
        escaped = re.escape(skill)
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        hits = len(pattern.findall(text))
        if hits > 0:
            found[skill] = hits

    # Add a few common technical skills often present in resumes but not necessarily in ontology.
    common_terms = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "C++",
        "C#",
        "HTML",
        "CSS",
        "React",
        "Node",
        "Node.js",
        "Next.js",
        "Git",
        "REST",
        "API",
        "AWS",
        "GCP",
        "Azure",
        "MongoDB",
        "PostgreSQL",
        "MySQL",
    ]
    for term in common_terms:
        escaped = re.escape(term)
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        hits = len(pattern.findall(text))
        if hits > 0:
            found.setdefault(term, hits)

    def infer_parents(skill: str) -> list[str]:
        parents = _case_insensitive_ontology_lookup(ontology, skill)
        if parents is not None:
            return parents

        s = skill.casefold()
        if s in {"python", "java", "c++", "c#", "javascript", "typescript"}:
            return ["Programming Fundamentals"]
        if s in {"html", "css", "react", "next.js"}:
            return ["Frontend"]
        if s in {"node", "node.js", "rest", "api"}:
            return ["Backend"]
        if s in {"aws", "gcp", "azure"}:
            return ["Cloud"]
        if s in {"git"}:
            return ["Software Engineering"]
        if s in {"mongodb", "postgresql", "mysql"}:
            return ["Database"]
        return []

    def score_from_hits(skill: str, hits: int) -> int:
        # Very rough scoring without an LLM: more mentions => higher confidence.
        base = 4
        bonus = min(6, hits * 2)
        score = base + bonus
        return max(0, min(10, score))

    items: list[ExtractedSkill] = []
    for skill, hits in sorted(found.items(), key=lambda kv: (-kv[1], kv[0].casefold())):
        parents = infer_parents(skill)
        items.append(ExtractedSkill(skill=skill, parents=parents, score=score_from_hits(skill, hits)))

    # Keep output focused.
    return items[:40]


def llm_extract_skills_with_agent(
    resume_text: str,
    ontology: dict[str, list[str]],
    model_name: str,
) -> list[ExtractedSkill]:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    @tool
    def get_skill_ontology() -> str:
        """Return the full skill ontology as JSON mapping skill -> parents."""
        return json.dumps(ontology, ensure_ascii=False)

    system = (
        "You are a resume skill extraction agent. "
        "Given resume text, extract a list of concrete skills mentioned or strongly implied. "
        "For each skill, output an object with: "
        "skill (string), parents (list of parent skill names), score (integer 0-10). "
        "Use the ontology tool to map skills to parents where applicable; when a skill is not "
        "in the ontology, pick the closest parent categories from the ontology (or [] if none). "
        "Score guidance: 9-10 if used across multiple projects/roles with strong evidence; "
        "7-8 if used meaningfully in at least one project/role; 5-6 if listed as a skill or lightly used; "
        "1-4 if only mentioned in passing. "
        "Return ONLY valid JSON: an array of objects. No markdown, no commentary."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Resume text:\n\n{resume_text}\n\nRemember: output only JSON."),
        ]
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    tools = [get_skill_ontology]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({"resume_text": resume_text})
    raw = strip_code_fences(str(result.get("output", "")))

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("LLM output was not a JSON array")

    items: list[ExtractedSkill] = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        skill = str(obj.get("skill", "")).strip()
        if not skill:
            continue
        parents_val = obj.get("parents", [])
        if isinstance(parents_val, str):
            parents = [parents_val]
        elif isinstance(parents_val, list):
            parents = [str(p) for p in parents_val if str(p).strip()]
        else:
            parents = []
        try:
            score = int(obj.get("score", 0))
        except Exception:
            score = 0
        score = max(0, min(10, score))
        items.append(ExtractedSkill(skill=skill, parents=parents, score=score))

    # De-dup by skill name (case-insensitive), keep highest score.
    best: dict[str, ExtractedSkill] = {}
    for item in items:
        key = item.skill.casefold()
        prev = best.get(key)
        if prev is None or item.score > prev.score:
            best[key] = item

    return sorted(best.values(), key=lambda x: (-x.score, x.skill.casefold()))


def write_output_json(items: Iterable[ExtractedSkill], output_path: Path) -> None:
    payload: list[dict[str, Any]] = [
        {"skill": i.skill, "parents": i.parents, "score": i.score} for i in items
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract skills from a resume PDF into JSON.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("data/candidate_resume/Sanjana_Boggaram_Resume.pdf"),
        help="Path to resume PDF",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("data/skill_ontology.json"),
        help="Path to skill ontology JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/resume_skills.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Gemini via LangChain (requires GOOGLE_API_KEY in .env)",
    )

    args = parser.parse_args()

    load_dotenv()
    resume_path: Path = args.resume
    ontology_path: Path = args.ontology
    out_path: Path = args.out

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology not found: {ontology_path}")

    ontology = load_ontology(ontology_path)
    resume_text = normalize_text(extract_text_from_pdf(resume_path))

    items: list[ExtractedSkill]

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if args.use_llm:
        if not google_key:
            raise RuntimeError(
                "--use-llm was set but GOOGLE_API_KEY is missing. Put it in .env and re-run."
            )
        items = llm_extract_skills_with_agent(resume_text, ontology, model_name=model_name)
    else:
        items = heuristic_extract_skills(resume_text, ontology)

    write_output_json(items, out_path)
    print(f"Wrote {len(items)} skills to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
