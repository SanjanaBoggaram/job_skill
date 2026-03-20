#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

RESUME_PATH="${RESUME_PATH:-data/candidate_resume/Sanjana_Boggaram_Resume.pdf}"
LOCATION_QUERY="${LOCATION_QUERY:-none}"
ROLE_QUERY="${ROLE_QUERY:-none}"
TOP_K="${TOP_K:-5}"

# Resume extraction can run without an API key (heuristic mode).
# If OPENROUTER_API_KEY is available, we default to using LLM extraction.
USE_LLM="${USE_LLM:-auto}"   # auto | 1 | 0

MIN_GAP="${MIN_GAP:-0.01}"
MAX_GAPS="${MAX_GAPS:-25}"

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --resume PATH        Resume PDF path (default: data/candidate_resume/Sanjana_Boggaram_Resume.pdf)
  --location TEXT      Location filter (default: none)
  --role TEXT          Role query for semantic filtering (default: none)
  --top-k N            Top-K jobs after role similarity (default: 5)
  --use-llm            Force LLM resume extraction
  --no-llm             Force heuristic resume extraction
  --min-gap X          Report: keep only gaps with score > X (default: 0.01)
  --max-gaps N         Report: max gaps to process (default: 25)
  -h, --help           Show help

Environment variables (optional alternatives to flags):
  PYTHON_BIN, RESUME_PATH, LOCATION_QUERY, ROLE_QUERY, TOP_K, USE_LLM, MIN_GAP, MAX_GAPS

Notes:
  - This script assumes job postings have already been processed and that outputs/job_postings_index.json exists.
  - filter/report steps require OPENROUTER_API_KEY (loaded from .env if present).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_PATH="$2"; shift 2 ;;
    --location)
      LOCATION_QUERY="$2"; shift 2 ;;
    --role)
      ROLE_QUERY="$2"; shift 2 ;;
    --top-k)
      TOP_K="$2"; shift 2 ;;
    --use-llm)
      USE_LLM="1"; shift ;;
    --no-llm)
      USE_LLM="0"; shift ;;
    --min-gap)
      MIN_GAP="$2"; shift 2 ;;
    --max-gaps)
      MAX_GAPS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

mkdir -p outputs

if [[ ! -f "outputs/job_postings_index.json" ]]; then
  echo "Missing outputs/job_postings_index.json" >&2
  echo "Run the job-postings processing + embedding/indexing scripts first." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Missing OPENROUTER_API_KEY (set it in .env)." >&2
  echo "This is required for filtering (embeddings) and report generation." >&2
  exit 1
fi

echo "[1/4] Extracting resume skills..."
resume_args=(
  --resume "$RESUME_PATH"
  --ontology "data/skill_ontology.json"
  --out "outputs/resume_skills.json"
)

if [[ "$USE_LLM" == "1" ]]; then
  resume_args+=(--use-llm)
elif [[ "$USE_LLM" == "auto" ]]; then
  resume_args+=(--use-llm)
fi

"$PYTHON_BIN" scripts/extract_resume_skills.py "${resume_args[@]}"

echo "[2/4] Filtering jobs + averaging skills..."
"$PYTHON_BIN" scripts/filter_jobs_and_average.py \
  --index "outputs/job_postings_index.json" \
  --resume-skills "outputs/resume_skills.json" \
  --ontology "data/skill_ontology.json" \
  --out-filtered "outputs/filtered_job_postings.json" \
  --out-average "outputs/average_job_skills.json" \
  --location "$LOCATION_QUERY" \
  --role "$ROLE_QUERY" \
  --top-k "$TOP_K"

echo "[3/4] Generating gap skills..."
"$PYTHON_BIN" scripts/generate_gap_skills.py \
  --resume "outputs/resume_skills.json" \
  --average "outputs/average_job_skills.json" \
  --out "outputs/gap_skills.json"

echo "[4/4] Generating report + learning plan..."
"$PYTHON_BIN" scripts/report_generate.py \
  --gaps "outputs/gap_skills.json" \
  --resources "data/study_resources/resources" \
  --out-summary "outputs/gap_summary.md" \
  --out-selected "outputs/gap_selected_resources.json" \
  --out-plan "outputs/learning_plan.md" \
  --min-gap "$MIN_GAP" \
  --max-gaps "$MAX_GAPS"

echo "Done. Outputs:" 
echo "- outputs/resume_skills.json" 
echo "- outputs/filtered_job_postings.json" 
echo "- outputs/average_job_skills.json" 
echo "- outputs/gap_skills.json" 
echo "- outputs/gap_summary.md" 
echo "- outputs/gap_selected_resources.json" 
echo "- outputs/learning_plan.md"
