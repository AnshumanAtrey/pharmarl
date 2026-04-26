"""Visual helpers for the PharmaRL Gradio UI.

Pure functions — no Gradio coupling. Take a SMILES (or list of them) and
return SVG strings, property dicts, and Plotly figures the app embeds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw, Lipinski, QED
from rdkit.Chem.Draw import rdMolDraw2D


_MOL_DRAW_OPTS = rdMolDraw2D.MolDrawOptions()
_MOL_DRAW_OPTS.bondLineWidth = 2
_MOL_DRAW_OPTS.padding = 0.05
_MOL_DRAW_OPTS.baseFontSize = 0.7
_MOL_DRAW_OPTS.setBackgroundColour((0.117, 0.149, 0.184, 1.0))  # slate-900-ish
_MOL_DRAW_OPTS.setSymbolColour((0.92, 0.96, 0.99, 1.0))         # slate-50
_MOL_DRAW_OPTS.setAtomPalette({
    0: (0.92, 0.96, 0.99, 1.0),    # default
    6: (0.92, 0.96, 0.99, 1.0),    # C
    7: (0.40, 0.78, 0.95, 1.0),    # N — sky
    8: (0.95, 0.42, 0.42, 1.0),    # O — coral
    9: (0.40, 0.95, 0.78, 1.0),    # F — emerald
    16: (0.95, 0.85, 0.40, 1.0),   # S — amber
    17: (0.40, 0.95, 0.50, 1.0),   # Cl — green
})


def mol_to_svg(smiles: str, size: tuple[int, int] = (340, 240)) -> str:
    """Render a SMILES to an inline-embeddable SVG. Returns a placeholder
    box if the SMILES doesn't parse so the UI never breaks on an invalid
    intermediate state."""
    if not smiles:
        return _empty_svg(*size, "no molecule yet")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return _empty_svg(*size, "invalid SMILES")
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.SetDrawOptions(_MOL_DRAW_OPTS)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _empty_svg(w: int, h: int, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="#1e293b" rx="12"/>'
        f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#64748b" font-family="ui-monospace, monospace" font-size="13">'
        f'{label}</text></svg>'
    )


def mol_properties(smiles: str) -> Dict[str, float | int | bool | str]:
    """Compute the chemist-facing properties: MW, LogP, HBD, HBA, QED,
    rotatable bonds, Lipinski pass/fail. Tolerant of invalid SMILES."""
    if not smiles:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    qed = QED.qed(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    return {
        "smiles": smiles,
        "mw": round(mw, 1),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "qed": round(qed, 3),
        "rotatable_bonds": rot,
        "lipinski_violations": violations,
        "lipinski_pass": violations <= 1,  # Ro5 allows 1 violation
    }


# Known reference drugs for the "compare to known drug" panel.
KNOWN_DRUGS: dict[str, dict[str, str]] = {
    "DRD2_dopamine_D2_receptor": {
        "name": "haloperidol",
        "smiles": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
        "note": "Antipsychotic; ~0.99 on the same DRD2 oracle PharmaRL uses.",
    },
    "GSK3B": {
        "name": "lithium",
        "smiles": "[Li+]",
        "note": "Mood stabilizer; canonical GSK3B inhibitor reference.",
    },
    "JNK3": {
        "name": "SP600125",
        "smiles": "O=c1[nH]c2cccc3cccc1c23",
        "note": "Pan-JNK inhibitor research tool; common JNK3 baseline.",
    },
}


def tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    """Tanimoto similarity over Morgan fingerprints (radius 2). Returns
    None if either SMILES is unparseable. Used for the 'how close to a
    known drug' gauge."""
    try:
        m_a = Chem.MolFromSmiles(smiles_a)
        m_b = Chem.MolFromSmiles(smiles_b)
        if m_a is None or m_b is None:
            return None
        fp_a = AllChem.GetMorganFingerprintAsBitVect(m_a, 2, 2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(m_b, 2, 2048)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return None


def reward_curve_figure(rewards: Sequence[float]):
    """Plotly line chart of cumulative reward over steps. Returns a Figure
    that gr.Plot can render directly."""
    import plotly.graph_objects as go

    cum = []
    s = 0.0
    for r in rewards:
        s += r
        cum.append(s)
    xs = list(range(1, len(rewards) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=cum, mode="lines+markers",
        line=dict(color="#10b981", width=3, shape="spline"),
        marker=dict(color="#10b981", size=8),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.12)",
        hovertemplate="step %{x}<br>cumulative %{y:+.3f}<extra></extra>",
        name="cumulative",
    ))
    if rewards:
        fig.add_trace(go.Bar(
            x=xs, y=list(rewards),
            marker_color=["#10b981" if r >= 0 else "#ef4444" for r in rewards],
            opacity=0.55,
            hovertemplate="step %{x}<br>reward %{y:+.3f}<extra></extra>",
            name="per-step",
        ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=10, b=30),
        height=260,
        showlegend=False,
        xaxis=dict(title="step", gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title="reward", gridcolor="rgba(148,163,184,0.15)"),
        hoverlabel=dict(font_family="ui-monospace, monospace"),
    )
    return fig


def reward_breakdown_figure(components: dict[str, float]):
    """Horizontal bar chart of the final reward components (binding, qed,
    sa, toxicity_clean). Each bar weighted by composite weight from the env."""
    import plotly.graph_objects as go

    # Display order + composite weights (matches server/grader.py defaults).
    order = [
        ("docking",          "binding (DRD2/TDC)", 0.40, "#06b6d4"),
        ("qed",              "drug-likeness (QED)",  0.25, "#10b981"),
        ("sa",               "synthesizability (SA)", 0.15, "#f59e0b"),
        ("toxicity_clean",   "non-toxic (CYP3A4⁻)",  0.20, "#a855f7"),
    ]
    rows = []
    for key, label, weight, color in order:
        v = components.get(key, 0.0) or 0.0
        rows.append((label, float(v), weight, color))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r[1] for r in rows],
        y=[r[0] for r in rows],
        orientation="h",
        marker_color=[r[3] for r in rows],
        text=[f"{r[1]:.2f} × w {r[2]:.2f}" for r in rows],
        textposition="outside",
        hovertemplate="%{y}<br>raw %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=20),
        height=200,
        xaxis=dict(range=[0, 1.15], gridcolor="rgba(148,163,184,0.15)",
                   tickformat=".1f", title="oracle score (0–1)"),
        yaxis=dict(automargin=True),
        showlegend=False,
    )
    return fig


def action_histogram_figure(action_counts: dict[str, int]):
    """Vertical bar chart of action-type usage across the episode."""
    import plotly.graph_objects as go

    order = ["ADD_FRAGMENT", "REMOVE_FRAGMENT", "SUBSTITUTE_ATOM", "TERMINATE"]
    colors = {"ADD_FRAGMENT": "#10b981", "REMOVE_FRAGMENT": "#ef4444",
              "SUBSTITUTE_ATOM": "#06b6d4", "TERMINATE": "#a855f7"}
    counts = [action_counts.get(k, 0) for k in order]

    fig = go.Figure(go.Bar(
        x=order, y=counts,
        marker_color=[colors[k] for k in order],
        text=counts, textposition="outside",
        hovertemplate="%{x}<br>used %{y}×<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=40),
        height=200,
        showlegend=False,
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(size=10)),
        yaxis=dict(title="count", gridcolor="rgba(148,163,184,0.15)"),
    )
    return fig
