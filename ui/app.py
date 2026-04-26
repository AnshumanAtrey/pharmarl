"""PharmaRL — Live Drug Discovery Operations Center.

Stripped-down dashboard: 3D molecule viewer, two-agent comparison
(pre-training baseline vs the H200-trained adapter), reward / oracle
breakdown for the selected agent.

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


def _load_dotenv() -> None:
    """Load pharmarl/.env into os.environ before anything else imports.

    The agents module checks env vars at import-time to decide whether
    each agent is live or static-card. Loading .env after that import
    would leave the agents permanently in static-card mode for this
    process, even if the env vars exist on disk.
    """
    env_path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

import gradio as gr  # noqa: E402

from models import MoleculeAction, MoleculeObservation  # noqa: E402,F401
from server.drug_discovery_environment import DrugDiscoveryEnvironment  # noqa: E402

from ui.agents import AGENTS, get_agent, make_agent  # noqa: E402
from ui.render import (  # noqa: E402
    KNOWN_DRUGS,
    mol_properties,
    mol_to_3d_html,
    reward_breakdown_figure,
    reward_curve_figure,
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
.gradio-container { max-width: 1240px !important; margin: 0 auto; }
.hero { padding: 22px 28px; border-radius: 18px;
        background: linear-gradient(135deg, #064e3b 0%, #0e7490 50%, #1e3a8a 100%);
        border: 1px solid rgba(16,185,129,0.35);
        box-shadow: 0 10px 40px -10px rgba(16,185,129,0.25); margin-bottom: 16px; }
.hero h1 { font-size: 30px !important; margin: 0 0 6px 0 !important;
           background: linear-gradient(135deg, #6ee7b7, #67e8f9, #c4b5fd);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           background-clip: text; font-weight: 800; }
.hero p  { margin: 4px 0 !important; color: #cbd5e1; }
.cum-card { padding: 22px 26px; border-radius: 16px;
            background: linear-gradient(135deg, #0f3027 0%, #134e4a 100%);
            border: 1px solid #10b981; }
.cum-label { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
             color: #6ee7b7; margin-bottom: 4px; }
.cum-value { font-size: 44px; font-weight: 800; font-family: ui-monospace, monospace;
             color: #ecfdf5; line-height: 1; }
.cum-sub   { font-size: 12px; color: #99f6e4; margin-top: 6px; font-family: ui-monospace, monospace; }
#trace textarea { font-family: 'JetBrains Mono', ui-monospace, monospace !important;
                  font-size: 12.5px !important; background: #0a0f1c !important;
                  color: #e2e8f0 !important; line-height: 1.55 !important; }
.section-title { font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase;
                 color: #94a3b8; font-weight: 600; margin: 4px 0 8px 0; }
.audit a { color: #6ee7b7; text-decoration: none; }
.audit a:hover { text-decoration: underline; }
.compare-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.compare-card { padding: 14px 16px; border-radius: 12px; border: 1px solid #1e293b;
                background: #0f172a; }
.compare-card.before { border-left: 3px solid #94a3b8; }
.compare-card.after  { border-left: 3px solid #10b981; }
.compare-card .num { font-size: 28px; font-weight: 700; font-family: ui-monospace, monospace;
                     color: #e2e8f0; line-height: 1; margin-top: 4px; }
.compare-card .lbl { font-size: 11px; color: #94a3b8; letter-spacing: 0.12em;
                     text-transform: uppercase; }
.compare-card .sub { font-size: 11px; color: #64748b; margin-top: 6px;
                     font-family: ui-monospace, monospace; }
"""


# ─── Episode runner ──────────────────────────────────────────────────────


def _format_step_line(step_num: int, action, reward: float, cum: float, msg: str) -> str:
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


def _cum_card(cum: float, max_seen: float, lipinski_ok: bool, label: str = "cumulative reward") -> str:
    badge = "Lipinski ✓" if lipinski_ok else "Lipinski ✗"
    return (
        f'<div class="cum-card">'
        f'<div class="cum-label">{label}</div>'
        f'<div class="cum-value">{cum:+.3f}</div>'
        f'<div class="cum-sub">max {max_seen:+.3f} · {badge}</div>'
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


def _baseline_card_explainer(info, target: str) -> tuple[str, str]:
    """Return (trace_text, mol_html) for a non-live baseline-card agent.

    Renders the published reference-drug structure for the chosen target
    so the molecule pane is never empty even before any live agent runs.
    """
    lines = [f"=== {info.label} ===\n"]
    lines.append(f"📚 baseline card — not a live policy in this UI.\n")
    lines.append(f"{info.blurb}\n")
    if info.baseline is not None:
        lines.append(f"\nPublished mean reward: {info.baseline:+.3f}\n")
    drug = KNOWN_DRUGS.get(target)
    if drug:
        lines.append(f"\nShowing the reference drug for {target.split('_')[0]}: {drug['name']}.")
        lines.append(f"  SMILES: {drug['smiles']}")
        lines.append(f"  {drug['note']}")
        mol_html = mol_to_3d_html(drug["smiles"])
    else:
        mol_html = mol_to_3d_html("")
    return "\n".join(lines), mol_html


def run_episode(difficulty: str, agent_key: str, target: str, max_steps: int) -> Generator[Tuple, None, None]:
    """Yields (trace, mol_html, props_md, cum_card, reward_curve_fig,
    breakdown_fig, final_scores_md) per step. Both current agents are
    baseline-card stubs; the live trainer hookup will land later."""
    info = get_agent(agent_key)

    if not info.live:
        trace, mol_html = _baseline_card_explainer(info, target)
        cum_html = _cum_card(info.baseline or 0.0, info.baseline or 0.0, True,
                             label="published reward")
        # If we have a reference drug for the target, show its property table.
        drug = KNOWN_DRUGS.get(target)
        props = mol_properties(drug["smiles"]) if drug else {}
        empty_curve = reward_curve_figure([])
        empty_bd = reward_breakdown_figure({})
        yield (trace, mol_html, _properties_md(props), cum_html,
               empty_curve, empty_bd, "_baseline-only — no live run_")
        return

    agent = make_agent(agent_key, difficulty)
    env = DrugDiscoveryEnvironment(seed=42)
    obs = env.reset(difficulty=difficulty, target=target if target else None)

    trace = (f"=== {info.label} on {difficulty} (target={obs.target.split('_')[0]}) ===\n"
             f"start    | seed scaffold     | SMILES={obs.smiles}\n\n")
    rewards: list[float] = []
    cum = 0.0
    max_seen = 0.0
    final_components: dict[str, float] | None = None
    last_smiles = obs.smiles

    yield (trace, mol_to_3d_html(obs.smiles), _properties_md(mol_properties(obs.smiles)),
           _cum_card(0.0, 0.0, True),
           reward_curve_figure([]), reward_breakdown_figure({}),
           "_episode running…_")

    for step_num in range(1, max_steps + 1):
        try:
            action = agent.next_action(obs)
        except Exception as e:
            trace += f"step {step_num:>2} | AGENT ERROR: {e}\n"
            yield trace, mol_to_3d_html(last_smiles), _properties_md(mol_properties(last_smiles)), \
                  _cum_card(cum, max_seen, True), \
                  reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}), \
                  "_agent crashed_"
            return

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
                      f"composite:    {final_components.get('composite', cum):.3f}\n")
            break

        yield (trace, mol_to_3d_html(obs.smiles), _properties_md(mol_properties(obs.smiles)),
               _cum_card(cum, max_seen, _lipinski_ok(obs.smiles)),
               reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}),
               "_episode running…_")
        # Live LLM calls already take 1-3s per step — no cosmetic delay needed.

    yield (trace, mol_to_3d_html(last_smiles), _properties_md(mol_properties(last_smiles)),
           _cum_card(cum, max_seen, _lipinski_ok(last_smiles)),
           reward_curve_figure(rewards), reward_breakdown_figure(final_components or {}),
           _final_scores_md(final_components, obs.target))


def _lipinski_ok(smiles: str) -> bool:
    p = mol_properties(smiles)
    return bool(p and p.get("lipinski_pass", True))


# ─── Layout ───────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PharmaRL — Live Operations Center") as app:
        gr.HTML(
            '<div class="hero">'
            '<h1>💊 PharmaRL — Live Drug Discovery</h1>'
            '<p>OpenEnv-native molecular RL — chat-agent SMILES editing, '
            'TDC oracle reward, GRPO post-training on H200.</p>'
            '</div>'
        )

        with gr.Row():
            # ─── LEFT — controls + cumulative ────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="section-title">scenario</div>')
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

                # Show "live ✓" or "card" next to each agent so it's
                # obvious at a glance whether the env vars got picked up.
                _agent_choices = [
                    (f"{a.label}  ·  {'live ✓' if a.live else 'card'}", a.key)
                    for a in AGENTS
                ]
                gr.HTML('<div class="section-title">agent</div>')
                agent = gr.Radio(
                    choices=_agent_choices,
                    value="pretrained",
                    label="Pre-training vs post-GRPO",
                )

                max_steps = gr.Slider(1, 20, value=12, step=1, label="Max steps")
                run_btn = gr.Button("▶  Run episode", variant="primary", size="lg")

                gr.HTML('<div class="section-title">cumulative reward</div>')
                cum_card = gr.HTML(_cum_card(0.0, 0.0, True))

            # ─── RIGHT — molecule + trace ────────────────────────────────
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">molecule (3D · drag to rotate)</div>')
                mol_html = gr.HTML(mol_to_3d_html(""))

                gr.HTML('<div class="section-title">properties</div>')
                props_md = gr.Markdown(_properties_md({}))

                gr.HTML('<div class="section-title">episode trace</div>')
                trace = gr.Textbox(
                    value="pick an agent + scenario, hit ▶ Run episode",
                    label="",
                    lines=10, max_lines=18,
                    elem_id="trace",
                    show_label=False,
                )

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-title">reward curve</div>')
                reward_plot = gr.Plot(reward_curve_figure([]), show_label=False)
            with gr.Column():
                gr.HTML('<div class="section-title">final reward decomposition</div>')
                breakdown_plot = gr.Plot(reward_breakdown_figure({}), show_label=False)

        gr.HTML('<div class="section-title" style="margin-top:14px"> </div>')
        final_md = gr.Markdown("")

        gr.HTML(
            '<div class="audit" style="margin-top:18px; padding:12px 16px; '
            'border-radius:12px; background:#0f172a; border:1px solid #1e293b; '
            'font-size:12.5px; color:#94a3b8;">'
            'audit trail · '
            '<a href="https://huggingface.co/vijay2776/pharmarl-llama-3b-trained-vijay-h200">'
            'HF Hub adapter + logs</a> · '
            '<a href="https://wandb.ai/vijaykota2776-itm/pharmarl/runs/zke7p0gr">W&amp;B run</a> · '
            '<a href="https://github.com/AnshumanAtrey/pharmarl/tree/vijay">GitHub branch</a>'
            '</div>'
        )

        run_btn.click(
            fn=run_episode,
            inputs=[difficulty, agent, target, max_steps],
            outputs=[trace, mol_html, props_md, cum_card,
                     reward_plot, breakdown_plot, final_md],
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
