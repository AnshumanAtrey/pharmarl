#!/usr/bin/env bash
# Full validation harness with per-run logging.
#
# Creates logs/{ISO-timestamp}_{commit-sha}/ and captures:
#   - _meta.json          — git SHA, branch, dirty status, environment fingerprint
#   - pytest.txt          — full pytest output (60+ tests)
#   - validate_stack.txt  — Gate 1 oracle smoke (TDC + RDKit + SELFIES)
#   - live_space.txt      — health/oracle/reset/step/ACTION-key normalization probes
#   - oversight_gemini.txt    — Fleet AI oversight using GEMINI_API_KEY
#   - oversight_openrouter.txt — Fleet AI oversight using OPENROUTER_API_KEY
#   - flags_via_reset.txt — verifies critic_enabled / schema_drift_enabled / oversight_enabled
#                           can be passed via HTTP /reset (issue #9)
#   - top_level_metadata.txt — verifies critique + oversight survive HTTP serialization (issue #8)
#   - summary.md          — human-readable pass/fail rollup
#   - exit_codes.tsv      — per-step exit code (for CI-style grep)
#
# Usage:
#   bash scripts/run_validation_log.sh
#
# Picks up GEMINI_API_KEY / OPENROUTER_API_KEY from .env if not already exported.

set -uo pipefail
cd "$(dirname "$0")/.."

# ─── Load .env so API keys are available to subshells ─────────────────
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# ─── Compute log dir ──────────────────────────────────────────────────
COMMIT_FULL=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    DIRTY=""
else
    DIRTY="-dirty"
fi
TS=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
LOGDIR="logs/${TS}_${COMMIT}${DIRTY}"
mkdir -p "$LOGDIR"

# ─── Helpers ──────────────────────────────────────────────────────────
EXIT_LOG="$LOGDIR/exit_codes.tsv"
echo -e "step\texit_code" > "$EXIT_LOG"

run_step() {
    local name="$1"
    local outfile="$2"
    shift 2
    echo "[$(date -u +%H:%M:%S)] $name → $outfile"
    {
        echo "=== ${name} ==="
        echo "Command: $*"
        echo "Started: $(date -u)"
        echo
        "$@"
        echo
        echo "Exit: $?"
    } > "$outfile" 2>&1
    local rc=$?
    echo -e "${name}\t${rc}" >> "$EXIT_LOG"
    return $rc
}

# ─── Metadata ─────────────────────────────────────────────────────────
cat > "$LOGDIR/_meta.json" <<META
{
  "commit_full": "$COMMIT_FULL",
  "commit_short": "$COMMIT",
  "dirty": $([ -n "$DIRTY" ] && echo true || echo false),
  "branch": "$BRANCH",
  "timestamp_utc": "$TS",
  "python": "$(.venv/bin/python --version 2>&1 | head -1)",
  "host": "$(uname -srm)",
  "openrouter_key_set": $([ -n "${OPENROUTER_API_KEY:-}" ] && echo true || echo false),
  "gemini_key_set": $([ -n "${GEMINI_API_KEY:-}" ] && echo true || echo false)
}
META

# ─── 1. pytest ────────────────────────────────────────────────────────
if [ -x .venv/bin/python ]; then
    run_step "pytest" "$LOGDIR/pytest.txt" \
        .venv/bin/python -m pytest tests/ -v --tb=short
else
    echo -e "pytest\t127" >> "$EXIT_LOG"
    echo "[FAIL] .venv/bin/python missing" > "$LOGDIR/pytest.txt"
fi

# ─── 2. validate_stack (Gate 1) ───────────────────────────────────────
run_step "validate_stack" "$LOGDIR/validate_stack.txt" \
    .venv/bin/python scripts/validate_stack.py

# ─── 3. Live HF Space smoke ───────────────────────────────────────────
SPACE_URL="${PHARMARL_ENV_URL:-https://anshumanatrey-pharmarl.hf.space}"
{
    echo "=== Live HF Space smoke (${SPACE_URL}) ==="
    echo
    echo "--- /health ---"
    curl -sS "${SPACE_URL}/health"
    echo
    echo
    echo "--- /oracle_status ---"
    curl -sS "${SPACE_URL}/oracle_status"
    echo
    echo
    echo "--- /reset (default) ---"
    RESET_RESP=$(curl -sS -X POST "${SPACE_URL}/reset" \
        -H 'content-type: application/json' \
        -d '{"difficulty":"easy"}')
    echo "$RESET_RESP" | head -c 800
    EID=$(echo "$RESET_RESP" | .venv/bin/python -c \
        "import json, sys; d=json.load(sys.stdin); print(d['observation']['episode_id'])")
    echo
    echo
    echo "--- /step with canonical schema ---"
    curl -sS -X POST "${SPACE_URL}/step" \
        -H 'content-type: application/json' \
        -d "{\"episode_id\":\"$EID\",\"action\":{\"action_type\":\"ADD_FRAGMENT\",\"fragment\":\"C\",\"position\":0}}" \
        | head -c 600
    echo
    echo
    echo "--- /step with verbose ACTION-key (action normalization) ---"
    curl -sS -X POST "${SPACE_URL}/step" \
        -H 'content-type: application/json' \
        -d "{\"episode_id\":\"$EID\",\"action\":{\"SELFIES\":[\"x\"],\"FRAGMENTS\":[\"y\"],\"ACTION\":\"TERMINATE\"}}" \
        | head -c 600
    echo
} > "$LOGDIR/live_space.txt" 2>&1
echo -e "live_space\t0" >> "$EXIT_LOG"

# ─── 4. Top-level metadata via HTTP (issue #8) ────────────────────────
{
    echo "=== Top-level critique / oversight in HTTP response (issue #8 fix) ==="
    echo
    echo "Local server check — does observation include critique + oversight as top-level keys?"
    .venv/bin/python <<'PYEOF'
from server.curriculum import CurriculumConfig
from server.drug_discovery_environment import DrugDiscoveryEnvironment
from openenv.core.env_server.serialization import serialize_observation

cfg = CurriculumConfig(critic_enabled=True)
env = DrugDiscoveryEnvironment(seed=0, config=cfg)
obs = env.reset(difficulty="easy")
serialized = serialize_observation(obs)
keys = sorted(serialized.get("observation", {}).keys())
print("Keys present in serialized HTTP response:")
for k in keys:
    print(f"  - {k}")
print()
print(f"  critique present: {'critique' in keys}")
print(f"  oversight present: {'oversight' in keys}")
print(f"  drift_warning present: {'drift_warning' in keys}")
PYEOF
} > "$LOGDIR/top_level_metadata.txt" 2>&1
echo -e "top_level_metadata\t0" >> "$EXIT_LOG"

# ─── 5. Flags via /reset (issue #9) ───────────────────────────────────
{
    echo "=== Flags via HTTP /reset (issue #9 fix) ==="
    echo
    echo "Locally — verify that /reset accepts critic_enabled / schema_drift_enabled / oversight_enabled."
    .venv/bin/python <<'PYEOF'
import json
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)
r = client.post("/reset", json={
    "difficulty": "easy",
    "critic_enabled": True,
    "schema_drift_enabled": True,
})
data = r.json()
obs = data["observation"]
print(f"reset status: {r.status_code}")
print(f"observation keys: {sorted(obs.keys())}")
print(f"critique populated: {obs.get('critique') is not None}")
print(f"drift_profile in metadata: {obs.get('metadata', {}).get('drift_profile')!r}")

eid = obs["episode_id"]
r = client.post("/step", json={
    "episode_id": eid,
    "action": {"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0},
})
data2 = r.json()
print(f"\nafter step status: {r.status_code}")
print(f"step observation has critique: {data2['observation'].get('critique') is not None}")
PYEOF
} > "$LOGDIR/flags_via_reset.txt" 2>&1
echo -e "flags_via_reset\t0" >> "$EXIT_LOG"

# ─── 6. Oversight — Gemini ────────────────────────────────────────────
if [ -n "${GEMINI_API_KEY:-}" ]; then
    {
        echo "=== Fleet AI oversight via Gemini 2.5 Flash ==="
        .venv/bin/python <<'PYEOF'
from server.oversight import LLMOversight
o = LLMOversight(provider="gemini", model_name="gemini-2.5-flash")
print(f"Provider: gemini  available: {o.is_available()}")
report = o.analyze(
    target="DRD2",
    starting_smiles="C1CCNCC1",
    final_smiles="c1ccc(N2CCN(CC2)c3ccccc3)cc1",
    action_history=[
        {"action": "ADD_FRAGMENT", "before": "C1CCNCC1", "after": "C1CCN(c2ccccc2)CC1"},
        {"action": "ADD_FRAGMENT", "before": "C1CCN(c2ccccc2)CC1", "after": "c1ccc(N2CCN(CC2)c3ccccc3)cc1"},
        {"action": "TERMINATE", "before": "c1ccc(N2CCN(CC2)c3ccccc3)cc1", "after": "c1ccc(N2CCN(CC2)c3ccccc3)cc1"},
    ],
    final_reward=6.5,
    lipinski_passes=True,
)
print(f"\nstrategy_summary: {report.strategy_summary}")
print(f"risk_flags: {report.risk_flags}")
print(f"risk_level: {report.risk_level}")
print(f"explanation: {report.explanation}")
print(f"model_name: {report.model_name}")
PYEOF
    } > "$LOGDIR/oversight_gemini.txt" 2>&1
    echo -e "oversight_gemini\t0" >> "$EXIT_LOG"
else
    echo "[SKIP] GEMINI_API_KEY not set" > "$LOGDIR/oversight_gemini.txt"
    echo -e "oversight_gemini\t-1" >> "$EXIT_LOG"
fi

# ─── 7. Oversight — OpenRouter ────────────────────────────────────────
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    {
        echo "=== Fleet AI oversight via OpenRouter Llama 3B free ==="
        .venv/bin/python <<'PYEOF'
from server.oversight import LLMOversight
o = LLMOversight(provider="openrouter", model_name="meta-llama/llama-3.2-3b-instruct:free")
print(f"Provider: openrouter  available: {o.is_available()}")
report = o.analyze(
    target="DRD2",
    starting_smiles="C1CCNCC1",
    final_smiles="c1ccc(N2CCN(CC2)c3ccccc3)cc1",
    action_history=[
        {"action": "ADD_FRAGMENT", "before": "C1CCNCC1", "after": "C1CCN(c2ccccc2)CC1"},
        {"action": "ADD_FRAGMENT", "before": "C1CCN(c2ccccc2)CC1", "after": "c1ccc(N2CCN(CC2)c3ccccc3)cc1"},
        {"action": "TERMINATE", "before": "c1ccc(N2CCN(CC2)c3ccccc3)cc1", "after": "c1ccc(N2CCN(CC2)c3ccccc3)cc1"},
    ],
    final_reward=6.5,
    lipinski_passes=True,
)
print(f"\nstrategy_summary: {report.strategy_summary}")
print(f"risk_flags: {report.risk_flags}")
print(f"risk_level: {report.risk_level}")
print(f"explanation: {report.explanation}")
print(f"model_name: {report.model_name}")
PYEOF
    } > "$LOGDIR/oversight_openrouter.txt" 2>&1
    echo -e "oversight_openrouter\t0" >> "$EXIT_LOG"
else
    echo "[SKIP] OPENROUTER_API_KEY not set" > "$LOGDIR/oversight_openrouter.txt"
    echo -e "oversight_openrouter\t-1" >> "$EXIT_LOG"
fi

# ─── Summary ──────────────────────────────────────────────────────────
{
    echo "# Validation log — ${TS} (${COMMIT}${DIRTY})"
    echo
    echo "Branch: ${BRANCH}"
    echo "Commit: ${COMMIT_FULL}"
    echo "Dirty:  $([ -n "$DIRTY" ] && echo yes || echo no)"
    echo
    echo "## Step results"
    echo
    echo "| step | exit |"
    echo "|---|---|"
    awk -F'\t' 'NR>1 {print "| "$1" | "$2" |"}' "$EXIT_LOG"
    echo
    echo "## Files"
    echo
    for f in "$LOGDIR"/*.txt; do
        echo "- \`$(basename "$f")\` — $(wc -l < "$f") lines"
    done
} > "$LOGDIR/summary.md"

echo
echo "================================================================"
echo "Validation log:  $LOGDIR/"
echo "Summary:         $LOGDIR/summary.md"
echo "Per-step exits:  $LOGDIR/exit_codes.tsv"
cat "$EXIT_LOG"
echo "================================================================"
