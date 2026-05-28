"""
HTML report from one Optuna study.

Sections (in this order):

1. Header: study name, ``n_trials``, best value, ceilings, timestamp.
2. Best params table with certificate ratio ``LMI_ceiling / tuned_value``
   per row.
3. Optimisation history (trial value vs trial number).
4. Per-scenario cost trace (``cost_J`` per scenario, per trial).
5. Param importance (Optuna fANOVA when enough trials).
6. Search-space coverage: 2-D scatter for the two most important params.

All plots use Plotly and render to a single self-contained HTML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import optuna
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tuning._types import Ceilings


_TEMPLATE_DIR = (Path(__file__).resolve().parent / "templates").resolve()


def _fig_history(study: optuna.Study) -> str:
    vals = [t.value for t in study.trials if t.value is not None]
    nums = [t.number for t in study.trials if t.value is not None]
    if not vals:
        return "<p>No completed trials.</p>"
    best_so_far = np.minimum.accumulate(vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nums, y=vals, mode="markers",
                             name="trial value"))
    fig.add_trace(go.Scatter(x=nums, y=list(best_so_far), mode="lines",
                             name="best so far"))
    fig.update_layout(
        xaxis_title="trial #", yaxis_title="CVaR(J)",
        title="Optimisation history",
        template="plotly_white", height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _fig_per_scenario(study: optuna.Study) -> str:
    if not study.trials:
        return "<p>No trials.</p>"
    scenario_keys = sorted({
        k.removeprefix("J__")
        for t in study.trials
        for k in t.user_attrs
        if k.startswith("J__")
    })
    if not scenario_keys:
        return "<p>No per-scenario diagnostics recorded.</p>"
    fig = go.Figure()
    any_data = False
    for sc in scenario_keys:
        ys = [t.user_attrs.get(f"J__{sc}") for t in study.trials]
        xs = [t.number for t in study.trials]
        finite_pairs = [
            (x, y) for x, y in zip(xs, ys)
            if y is not None and np.isfinite(y)
        ]
        if finite_pairs:
            any_data = True
            xs_f, ys_f = zip(*finite_pairs)
            fig.add_trace(go.Scatter(
                x=list(xs_f), y=list(ys_f),
                mode="lines+markers", name=sc,
            ))
    if not any_data:
        return "<p>No finite per-scenario costs recorded.</p>"
    fig.update_layout(
        xaxis_title="trial #", yaxis_title="cost_J",
        title="Per-scenario cost across trials",
        template="plotly_white", height=400, yaxis_type="log",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _fig_importance(study: optuna.Study) -> str:
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        return f"<p>Importance unavailable: {e}</p>"
    if not imp:
        return "<p>No importance data yet.</p>"
    names, vals = zip(*imp.items())
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(names), orientation="h",
    ))
    fig.update_layout(
        title="Parameter importance (fANOVA)",
        xaxis_title="importance",
        template="plotly_white", height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _fig_top2_scatter(study: optuna.Study) -> str:
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}
    if len(imp) < 2:
        return "<p>Need at least 2 params with importance data.</p>"
    p1, p2 = list(imp.keys())[:2]
    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for t in study.trials:
        if t.value is None:
            continue
        a, b = t.params.get(p1), t.params.get(p2)
        if a is None or b is None:
            continue
        xs.append(float(a)); ys.append(float(b)); cs.append(float(t.value))
    if not xs:
        return "<p>No trials with both params present.</p>"
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            size=8, color=cs, colorscale="Viridis",
            colorbar=dict(title="CVaR(J)"), showscale=True,
        ),
    ))
    fig.update_layout(
        xaxis_title=p1, yaxis_title=p2,
        title=f"Coverage: {p1} vs {p2}",
        xaxis_type="log", yaxis_type="log",
        template="plotly_white", height=500,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _certificate_ratios(
    best_params: dict[str, float],
    ceilings: Ceilings,
) -> List[dict[str, str]]:
    """Build the certificate-ratio table.  Ratio = ceiling / tuned."""
    cd = ceilings.as_dict()
    rows: List[dict[str, str]] = []
    for k, v in best_params.items():
        ceil = cd.get(k)
        ratio_str = "—"
        ceil_str = "—"
        if ceil is not None and np.isfinite(ceil):
            ceil_str = f"{ceil:.3g}"
            if v > 0.0:
                ratio_str = f"{ceil / v:.1f}×"
        rows.append({
            "param":   k,
            "tuned":   f"{v:.3g}",
            "ceiling": ceil_str,
            "ratio":   ratio_str,
        })
    return rows


def write_tuning_report(
    study: optuna.Study,
    ceilings: Ceilings,
    output: Path,
) -> None:
    """Render and write the tuning HTML report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("tuning_report.html.j2")

    best = study.best_trial
    rows = _certificate_ratios(best.params, ceilings)

    html = template.render(
        study_name=study.study_name,
        n_trials=len(study.trials),
        best_value=best.value if best.value is not None else float("nan"),
        best_trial=best.number,
        cert_rows=rows,
        ceilings_notes=ceilings.notes or "(no notes)",
        fig_history=_fig_history(study),
        fig_per_scenario=_fig_per_scenario(study),
        fig_importance=_fig_importance(study),
        fig_top2_scatter=_fig_top2_scatter(study),
    )
    output.write_text(html, encoding="utf-8")
