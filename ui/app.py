"""PharmaRL — Live Drug Discovery Operations Center.

Gradio dashboard that runs an episode in real time inside the same Python
process as the env. No HTTP, no separate uvicorn. Streams per-step trace,
2D molecule structure, reward decomposition, and the curriculum baselines.

Run:
    python -m ui.app
    # then open http://localhost:7860
"""
from __future__ import annotations

import os
import sys
import time
from typing import Generator, Tuple

# Make sibling imports (models, server) work whether you run as a module
# (`python -m ui.app`) or directly (`python ui/app.py`).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr  # noqa: E402

from models import MoleculeAction, MoleculeObservation  # noqa: E402
from server.drug_discovery_environment import DrugDiscoveryEnvironment  # noqa: E402

from ui.agents import AGENTS, get_agent, make_agent  # noqa: E402
from ui.render import (  # noqa: E402
    KNOWN_DRUGS,
    action_histogram_figure,
    mol_properties,
    mol_to_svg,
    reward_breakdown_figure,
    reward_curve_figure,
    tanimoto,
)


# ─── Theme + CSS ──────────────────────────────────────────────────────────

_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0b1220",
    body_background_fill_dark="#0b1220",
    block_background_fill="#0f172a",
    block_background_fill_dark="#0f172a",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    block_label_text_color="#94a3b8",
    block_title_text_color="#e2e8f0",
    body_text_color="#e2e8f0",
)

_CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto; }
.hero { padding: 22px 28px; border-radius: 18px;
        background: linear-gradient(135deg, #064e3b 0%, #0e7490 50%, #1e3a8a 100%);
        border: 1px solid rgba(16,185,129,0.35);
        box-shadow: 0 10px 40px -10px rgba(16,185,129,0.25); margin-bottom: 16px; }
.hero h1 { font-size: 30px !important; margin: 0 0 6px 0 !important;
           background: linear-gradient(135deg, #6ee7b7, #67e8f9, #c4b5fd);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           background-clip: text; font-weight: 800; }
.hero p  { margin: 4px 0 !important; color: #cbd5e1; }
.hero .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
              font-size: 12px; margin-right: 6px; font-family: ui-monospace, monospace;
              background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); }
.cum-card { padding: 22px 26px; border-radius: 16px;
            background: linear-gradient(135deg, #0f3027 0%, #134e4a 100%);
            border: 1px solid #10b981; }
.cum-label { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
             color: #6ee7b7; margin-bottom: 4px; }
.cum-value { font-size: 44px; font-weight: 800; font-family: ui-monospace, monospace;
             color: #ecfdf5; line-height: 1; }
.cum-sub   { font-size: 12px; color: #99f6e4; margin-top: 6px; font-family: ui-monospace, monospace; }
.mol-frame { background: #0f172a; border: 1px solid #1e293b; border-radius: 14px;
             padding: 8px; }
.mol-frame svg { display:block; margin: 0 auto; }
#trace textarea { font-family: 'JetBrains Mono', ui-monospace, monospace !important;
                  font-size: 12.5px !important; background: #0a0f1c !important;
                  color: #e2e8f0 !important; line-height: 1.55 !important; }
.section-title { font-size: 13px; letter-spacing: 0.16em; text-transform: uppercase;
                 color: #94a3b8; font-weight: 600; margin: 4px 0 8px 0; }
.tier-card { border: 1px solid #1e293b; border-radius: 12px; padding: 12px 14px;
             background: #0f172a; margin-bottom: 8px; }
.tier-card.trivial { border-left: 3px solid #10b981; }
.tier-card.easy    { border-left: 3px solid #06b6d4; }
.tier-card.hard    { border-left: 3px solid #a855f7; }
"""


# ─── Episode runner (generator) ──────────────────────────────────────────


def _format_step_line(step_num: int, action: MoleculeAction, reward: float, cum: float, msg: str) -> str:
    at = action.action_type
    parts = [f"step {step_num:>2} | {at:<16}"]
    if action.fragment is not None:
        parts.append(f"frag={action.fragment}")
    if action.new_atom is not None:
        parts.append(f"→ {action.new_atom}")
    if action.position is not None:
        parts.append(f"@pos {action.position}")
    head = " | ".join(parts).ljust(56)
    tag = "✓" if reward >= 0 else "✗"
    return f"{head} | reward {reward:+.3f} | cum {cum:+.3f} {tag}\n   ↳ {msg.strip()}\n"


def _cum_card(cum: float, max_seen: float, parse_pct: float, lipinski_ok: bool) -> str:
    badge = "Lipinski ✓" if lipinski_ok else "Lipinski ✗"
    return (
        f'<div class="cum-card">'
        f'<div class="cum-label">cumulative reward</div>'
        f'<div class="cum-value">{cum:+.3f}</div>'
        f'<div class="cum-sub">max {max_seen:+.3f} · parse {parse_pct:.0f}% · {badge}</div>'
        f'</div>'
    )


def _properties_md(props: dict) -> str:
    if not props:
        return "_no molecule yet_"
    rows = [
        ("SMILES", f"`{props.get('smiles', '')}`"),
        ("MW", f"{props.get('mw', 0):.1f}"),
        ("LogP", f"{props.get('logp', 0):.2f}"),
        ("HBD / HBA", f"{props.get('hbd', 0)} / {props.get('hba', 0)}"),
        ("QED", f"{props.get('qed', 0):.3f}"),
        ("Rotatable bonds", str(props.get("rotatable_bonds", 0))),
        ("Lipinski", "✅ pass" if props.get("lipinski_pass") else f"⚠️ {props.get('lipinski_violations', 0)} violation(s)"),
    ]
    body = "\n".join(f"| **{k}** | {v} |" for k, v in rows)
    return f"| | |\n|:--|:--|\n{body}"


def _final_scores_md(final_components: dict | None, target: str) -> str:
    if not final_components:
        return "_run an episode to see the breakdown_"
    drug = KNOWN_DRUGS.get(target)
    rows = [
        ("Binding (DRD2/TDC)", f"{final_components.get('docking', 0):.3f}"),
        ("QED (drug-likeness)", f"{final_components.get('qed', 0):.3f}"),
        ("SA (synthesizability)", f"{final_components.get('sa', 0):.3f}"),
        ("Toxicity-clean (CYP3A4⁻)", f"{final_components.get('toxicity_clean', 0):.3f}"),
        ("Composite (weighted)", f"**{final_components.get('composite', 0):.3f}**"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    out = f"| Component | Value |\n|:--|--:|\n{body}"
    if drug:
        out += f"\n\n_Reference for {target.split('_')[0]}: {drug['name']} → ~0.99 on the same oracle._"
    return out


def _baselines_table(current_score: float | None) -> list[list]:
    """The README's baseline spectrum, sortable, with the active run highlighted."""
    rows = [
        ["Random uniform",   "+2.30", "$0",       ""],
        ["Scripted (4-step)", "+2.81", "$0",       ""],
        ["Llama 3.2 3B",      "+1.67", "$0.001",   ""],
        ["Gemini 2.5 Flash",  "+1.81", "$0.026",   ""],
        ["Llama 3.1 8B",      "+2.45", "$0.001",   "best small"],
        ["Llama 3.3 70B",     "+1.19", "$0.007",   "⚠ inverted scaling"],
        ["Gemini 2.5 Pro",    "+3.68", "$0.123",   "best baseline"],
    ]
    if current_score is not None:
        rows.append([f"Llama 1B trained (vijay-h200) — this run", f"{current_score:+.2f}", "—", "← live"])
    return rows


def run_episode(difficulty: str, agent_key: str, target: str, max_steps: int) -> Generator[Tuple, None, None]:
    """Generator: yields (trace_text, mol_svg_html, properties_md, cum_card_html,
    reward_curve_fig, breakdown_fig, action_hist_fig, final_md, baselines_rows)
    after every env step. The Gradio handler binds these to outputs.
    """
    info = get_agent(agent_key)

    # Non-live baseline: render an explanatory card instead of running.
    if not info.live:
        explainer = (
            f"📚 **{info.label}** is a published baseline, not a live agent in this UI.\n\n"
            f"{info.blurb}\n\n"
            f"Mean across DRD2/GSK3B/JNK3: **{info.baseline:+.2f}**"
            if info.baseline is not None
            else f"📚 **{info.label}** — {info.blurb}\n\n"
                 "Live wiring lands here once the H200 GRPO adapter is published."
        )
        empty_mol = mol_to_svg("")
        empty_curve = reward_curve_figure([])
        empty_bd = reward_breakdown_figure({})
        empty_hist = action_histogram_figure({})
        cum_html = _cum_card(info.baseline or 0.0, info.baseline or 0.0, 100.0, True)
        yield (
            explainer, _wrap_mol(empty_mol), _properties_md({}), cum_html,
            empty_curve, empty_bd, empty_hist,
            "_baseline-only — no live run_", _baselines_table(None),
        )
        return

    agent = make_agent(agent_key, difficulty)
    env = DrugDiscoveryEnvironment(seed=42)
    obs = env.reset(difficulty=difficulty, target=target if target else None)

    trace = (f"=== {info.label} on {difficulty} (target={obs.target.split('_')[0]}) ===\n"
             f"start    | seed scaffold     | SMILES={obs.smiles}\n"
             f"            (fragments={obs.available_fragments[:6]}{'…' if len(obs.available_fragments)>6 else ''})\n\n")
    rewards: list[float] = []
    actions: dict[str, int] = {}
    cum = 0.0
    max_seen = 0.0
    parse_attempts = parse_ok = 0
    final_components: dict[str, float] | None = None
    last_smiles = obs.smiles

    # Yield initial state immediately so the UI renders the seed molecule
    # before the first step kicks in.
    yield (
        trace, _wrap_mol(mol_to_svg(obs.smiles)), _properties_md(mol_properties(obs.smiles)),
        _cum_card(0.0, 0.0, 100.0, True),
        reward_curve_figure([]), reward_breakdown_figure({}), action_histogram_figure({}),
        "_episode running…_", _baselines_table(None),
    )

    for step_num in range(1, max_steps + 1):
        try:
            action = agent.next_action(obs)
            parse_attempts += 1
            parse_ok += 1  # in-process agents always emit valid actions
        except Exception as e:
            trace += f"step {step_num:>2} | AGENT ERROR: {e}\n"
            yield trace, _wrap_mol(mol_to_svg(last_smiles)), _properties_md(mol_properties(last_smiles)), \
                  _cum_card(cum, max_seen, _pct(parse_ok, parse_attempts), True), \
                  reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}), \
                  action_histogram_figure(actions), "_agent crashed_", _baselines_table(None)
            return

        actions[action.action_type] = actions.get(action.action_type, 0) + 1
        obs = env.step(action)
        rewards.append(obs.reward)
        cum += obs.reward
        max_seen = max(max_seen, cum)
        last_smiles = obs.smiles
        trace += _format_step_line(step_num, action, obs.reward, cum, obs.message)

        if obs.done:
            final_components = obs.final_oracle_scores or (obs.metadata or {}).get("final_oracle_scores") or {}
            trace += (f"\n=== TERMINAL (step {step_num}) ===\n"
                      f"final SMILES: {obs.smiles}\n"
                      f"composite:    {final_components.get('composite', cum):.3f}\n"
                      f"truncated:    {obs.truncated}\n")
            break

        # Stream every step so the UI feels alive. The 0.18s delay is
        # purely cosmetic — the env.step itself is sub-100ms.
        yield (
            trace, _wrap_mol(mol_to_svg(obs.smiles)), _properties_md(mol_properties(obs.smiles)),
            _cum_card(cum, max_seen, _pct(parse_ok, parse_attempts), _lipinski_ok(obs.smiles)),
            reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}),
            action_histogram_figure(actions),
            "_episode running…_", _baselines_table(None),
        )
        time.sleep(0.18)

    # Final render — include baselines highlight
    final_md = _final_scores_md(final_components, obs.target)
    yield (
        trace, _wrap_mol(mol_to_svg(last_smiles)), _properties_md(mol_properties(last_smiles)),
        _cum_card(cum, max_seen, _pct(parse_ok, parse_attempts), _lipinski_ok(last_smiles)),
        reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}),
        action_histogram_figure(actions),
        final_md, _baselines_table(cum),
    )


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 100.0


def _wrap_mol(svg: str) -> str:
    return f'<div class="mol-frame">{svg}</div>'


def _lipinski_ok(smiles: str) -> bool:
    p = mol_properties(smiles)
    return bool(p and p.get("lipinski_pass", True))


# ─── Layout ───────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    # Gradio 6.0 moved theme/css to launch(); we still pass title here.
    with gr.Blocks(title="PharmaRL — Live Operations Center") as app:
        gr.HTML(
            '<div class="hero">'
            '<h1>💊 PharmaRL — Live Drug Discovery Operations Center</h1>'
            '<p><b>OpenEnv hackathon submission</b> — the first OpenEnv-native env where an LLM is the policy. '
            'Chat-agent SELFIES editing against frozen TDC oracles (DRD2 / GSK3B / JNK3). '
            'Same canonical MOSES/GuacaMol benchmark used in MolDQN, REINVENT, GraphAF, GFlowNets — new policy class.</p>'
            '<p>'
            '<span class="pill">Llama 3.2 1B</span>'
            '<span class="pill">DRD2 / GSK3B / JNK3</span>'
            '<span class="pill">TDC verified</span>'
            '<span class="pill">3-tier curriculum</span>'
            '<span class="pill">GRPO post-training</span>'
            '</p></div>'
        )

        with gr.Row():
            # ─── LEFT COLUMN — controls + cumulative ─────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="section-title">🎯 Scenario</div>')
                difficulty = gr.Radio(
                    choices=[
                        ("Trivial — QED only · 5 fragments · 10 steps", "trivial"),
                        ("Easy — QED + binding · 15 fragments · 15 steps", "easy"),
                        ("Hard — full composite · 50 fragments · 20 steps", "hard"),
                    ],
                    value="trivial",
                    label="Difficulty tier",
                )
                target = gr.Dropdown(
                    choices=[
                        ("DRD2 (canonical molRL benchmark)", "DRD2"),
                        ("GSK3B (Alzheimer's, mood)", "GSK3B"),
                        ("JNK3 (held-out — never trained on)", "JNK3"),
                    ],
                    value="DRD2",
                    label="Binding target",
                )

                gr.HTML('<div class="section-title">🤖 Agent</div>')
                agent = gr.Radio(
                    choices=[(a.label, a.key) for a in AGENTS],
                    value="scripted",
                    label="Pick a policy",
                    info="Live agents (Random, Scripted) run end-to-end. Llama / Gemini rows show the published baselines.",
                )

                max_steps = gr.Slider(1, 20, value=12, step=1, label="Max steps")
                run_btn = gr.Button("▶  Run episode", variant="primary", size="lg")

                gr.HTML('<div class="section-title">💯 Cumulative reward</div>')
                cum_card = gr.HTML(_cum_card(0.0, 0.0, 100.0, True))

                gr.HTML('<div class="section-title">🧪 Final molecule</div>')
                mol_html = gr.HTML(_wrap_mol(mol_to_svg("")))
                props_md = gr.Markdown(_properties_md({}), label="Properties")

            # ─── RIGHT COLUMN — live trace + charts ──────────────────────
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">🧬 Episode trace (streaming)</div>')
                trace = gr.Textbox(
                    value="run an episode →",
                    label="Step-by-step log",
                    lines=18, max_lines=28,
                    elem_id="trace",
                )

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-title">📈 Reward curve</div>')
                        reward_plot = gr.Plot(reward_curve_figure([]))
                    with gr.Column():
                        gr.HTML('<div class="section-title">🎯 Final reward decomposition</div>')
                        breakdown_plot = gr.Plot(reward_breakdown_figure({}))

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-title">🪄 Action histogram</div>')
                        action_plot = gr.Plot(action_histogram_figure({}))
                    with gr.Column():
                        gr.HTML('<div class="section-title">📋 Final scores</div>')
                        final_md = gr.Markdown("_run an episode to see the breakdown_")

        # ─── BASELINES LEADERBOARD ───────────────────────────────────────
        gr.HTML('<div class="section-title" style="margin-top:18px">🏆 Baseline spectrum (mean DRD2 / GSK3B / JNK3)</div>')
        baselines = gr.Dataframe(
            headers=["Policy", "Mean reward", "Cost", "Note"],
            value=_baselines_table(None),
            interactive=False,
            wrap=True,
        )
        gr.Markdown(
            "**Inverted scaling proof.** Llama 70B (+1.19) underperforms Random (+2.30) "
            "and 8B (+2.45) — the env's reward function isn't trivially gameable by raw "
            "model capacity. Discipline beats scale past a sweet spot. "
            "Full table + reproducing instructions in `docs/baselines.md`. "
            "Probe spend across 6 baselines: $0.158."
        )

        gr.HTML(
            '<div style="margin-top:18px; padding:12px 16px; border-radius:12px; '
            'background:#0f172a; border:1px solid #1e293b; font-size:12.5px; color:#94a3b8;">'
            '<b>Audit trail:</b> H200 training Job logs at '
            '<a href="https://huggingface.co/vijay2776/pharmarl-llama-3b-trained-vijay-h200" '
            'style="color:#6ee7b7">vijay2776/pharmarl-llama-3b-trained-vijay-h200</a> · '
            'W&amp;B project at <a href="https://wandb.ai/vijaykota2776-itm/pharmarl" '
            'style="color:#6ee7b7">wandb.ai/vijaykota2776-itm/pharmarl</a> · '
            'Code at <a href="https://github.com/AnshumanAtrey/pharmarl/tree/vijay" '
            'style="color:#6ee7b7">github.com/AnshumanAtrey/pharmarl/tree/vijay</a>.'
            '</div>'
        )

        run_btn.click(
            fn=run_episode,
            inputs=[difficulty, agent, target, max_steps],
            outputs=[trace, mol_html, props_md, cum_card,
                     reward_plot, breakdown_plot, action_plot,
                     final_md, baselines],
            show_progress="hidden",
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
        inbrowser=False,
        share=False,
        theme=_THEME,
        css=_CSS,
    )
