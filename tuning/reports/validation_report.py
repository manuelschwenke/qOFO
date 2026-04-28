"""
Validation report from a list of :class:`RunResult` on the validation
set.

Sections:

1. Header: tuned-params summary, ``n_scenarios``, PF-failure rate, J
   stats.
2. Cost histogram (``cost_J`` across scenarios, log-scale x).
3. Empirical contraction percentile histogram (``rho_emp_p95``).
4. Per-metric histograms: ITAE_v_TS, ITAE_v_DS, ITAE_q_pcc,
   n_osc_total, n_tap_switches_total.
5. Failure breakdown table: scenarios where ``pf_failures > 0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tuning.runner import RunResult


_TEMPLATE_DIR = (Path(__file__).resolve().parent / "templates").resolve()


def _hist(
    values: List[float],
    title: str,
    xlabel: str,
    log_x: bool = False,
    include_plotlyjs: bool | str = False,
) -> str:
    finite = [float(v) for v in values
              if v is not None and np.isfinite(v)]
    if not finite:
        return f"<p>No finite data for {title}.</p>"
    fig = go.Figure(go.Histogram(x=finite, nbinsx=40))
    layout: dict[str, Any] = dict(
        title=title, xaxis_title=xlabel, yaxis_title="count",
        template="plotly_white", height=350,
    )
    if log_x:
        layout["xaxis_type"] = "log"
    fig.update_layout(**layout)
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _failure_table(results: List[RunResult]) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for r in results:
        if r.metrics.pf_failures > 0 or r.failure_reason:
            rows.append({
                "scenario":    r.scenario_name,
                "pf_failures": int(r.metrics.pf_failures),
                "cost_J":      f"{r.metrics.cost_J:.3g}",
                "reason":      (r.failure_reason or "")[:200],
            })
    return rows


def write_validation_report(
    results: List[RunResult],
    params: dict[str, float],
    meta: dict[str, Any],
    output: Path,
) -> None:
    """Render and write the validation HTML report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("validation_report.html.j2")

    Js = [float(r.metrics.cost_J) for r in results]
    rhos = [float(r.metrics.rho_emp_p95) for r in results]
    n_osc_total = [
        int(r.metrics.n_osc_der + r.metrics.n_osc_pcc + r.metrics.n_osc_v_gen)
        for r in results
    ]
    n_tap_total = [
        int(r.metrics.n_tap_switches_tso + r.metrics.n_tap_switches_dso)
        for r in results
    ]

    if Js:
        median_J = float(np.median(Js))
        mean_J   = float(np.mean(Js))
        p95_J    = float(np.percentile(Js, 95))
        max_J    = float(np.max(Js))
    else:
        median_J = mean_J = p95_J = max_J = float("nan")

    if rhos:
        median_rho = float(np.median(rhos))
        p95_rho    = float(np.percentile(rhos, 95))
    else:
        median_rho = p95_rho = float("nan")

    html = template.render(
        n_scenarios=len(results),
        n_pf_fail=sum(1 for r in results if r.metrics.pf_failures > 0),
        n_runner_err=sum(1 for r in results if r.failure_reason),
        params=params,
        meta=meta,
        median_J=median_J, mean_J=mean_J, p95_J=p95_J, max_J=max_J,
        median_rho=median_rho, p95_rho=p95_rho,
        fig_J_hist=_hist(Js, "Cost J distribution", "cost_J",
                         log_x=True, include_plotlyjs="cdn"),
        fig_rho_hist=_hist(rhos,
                           "Empirical contraction (95th %ile per run)",
                           "rho_emp_p95"),
        fig_osc_hist=_hist([float(v) for v in n_osc_total],
                           "Oscillations per scenario", "n_osc_total"),
        fig_tap_hist=_hist([float(v) for v in n_tap_total],
                           "Tap switches per scenario", "n_tap_total"),
        fail_rows=_failure_table(results),
    )
    output.write_text(html, encoding="utf-8")
