# Validation log — post-redeploy fixes verified

**When:** 2026-04-26 03:13 UTC (~7 AM Bangalore)
**Commit:** `30260ee` (after `8e5c738` Sahil-bugs fix push and a teammate commit)
**Branch:** `main`
**HF Space build:** detected new code online at t+588s (~9.8 min after upload)

## What this proves

| Sub-theme prize | HTTP-reachable on live Space? |
|---|---|
| Patronus AI (schema drift) | ✅ `{"schema_drift_enabled":true,"drift_profile":"early_admet"}` mutates `active_constraints` |
| Halluminate (rules-based critic) | ✅ `{"critic_enabled":true}` populates top-level `obs.critique` |
| Fleet AI (LLM oversight) | ✅ `{"oversight_enabled":true}` triggers Gemini call at TERMINATE |

Plus:
- Issue #8 fixed: critique/oversight survive HTTP serialization (top-level fields, not metadata)
- Issue #9 fixed: per-episode flag overrides via `/reset` body
- Issue #10 fixed: `tick_labels=` (matplotlib 3.11-compatible)
- Issue #11 fixed: stale notebook-bug warning removed
- Issue #12 fixed: `setuptools<81` pinned for fresh-env compatibility

## Per-step results

| step | exit | what it covers |
|---|---|---|
| pytest | 0 | All 98 tests pass |
| validate_stack | 0 | TDC + RDKit + SELFIES + 4 oracles + multi-target trio |
| live_space | 0 | Live HF Space `/health`, `/oracle_status`, `/reset` (default + verbose), `/step` (canonical + ACTION-key normalized) |
| top_level_metadata | 0 | OpenEnv serializer surfaces `critique` + `oversight` + `drift_warning` as top-level keys |
| flags_via_reset | 0 | `/reset` accepts `critic_enabled`, `schema_drift_enabled`, `oversight_enabled`, `drift_profile` |
| oversight_gemini | 0 | Real Gemini 2.5 Flash call returned a structured oversight report |
| oversight_openrouter | 0 | OpenRouter free-tier 429-rate-limited (expected); graceful degradation |

## Real Gemini oversight output (from `oversight_gemini.txt`)

```
strategy_summary: The agent attempted to expand the initial piperidine ring
                  with phenyl fragments, resulting in a molecule with two
                  phenyl groups attached to a piperazine core.
risk_flags: []
risk_level: low
explanation: The trajectory shows a straightforward expansion strategy,
             resulting in a valid molecule that passes Lipinski's rules.
             No suspicious patterns or chemical anomalies were observed.
```

## Files in this directory

- `_meta.json` — git SHA, branch, dirty flag, Python version, host, API key fingerprint
- `pytest.txt` — full pytest output (98 tests)
- `validate_stack.txt` — TDC + RDKit + SELFIES + oracle stack health
- `live_space.txt` — live HF Space HTTP probes
- `top_level_metadata.txt` — issue #8 verification
- `flags_via_reset.txt` — issue #9 verification (using FastAPI TestClient)
- `oversight_gemini.txt` — Fleet AI oversight via Gemini 2.5 Flash
- `oversight_openrouter.txt` — Fleet AI oversight via OpenRouter Llama 3B
- `summary.md` — human-readable rollup
- `exit_codes.tsv` — CI-grep-friendly per-step status

## How to reproduce

```bash
bash scripts/run_validation_log.sh
```

Reads `GEMINI_API_KEY` and `OPENROUTER_API_KEY` from `.env` (or environment), creates a fresh `logs/{ISO-timestamp}_{commit-sha}/` directory, and runs the same 7 steps end-to-end.

`logs/` is in `.gitignore` — only this single snapshot is committed for shareability.
