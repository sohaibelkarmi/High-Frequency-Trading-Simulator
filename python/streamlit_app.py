"""Streamlit UI for exploring Hawkes simulations interactively."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .bridge_utils import ensure_bridge_path
    from .kernels import ExpKernel, PowerLawKernel
    from .timeline_dashboard import (
        SimulationTimeline,
        plot_timeline_interactive,
        simulate_exp_timeline,
        simulate_powerlaw_timeline,
    )
except (ImportError, ValueError):
    import pathlib
    import sys

    _pkg_dir = pathlib.Path(__file__).resolve().parent
    if str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

    from bridge_utils import ensure_bridge_path
    from kernels import ExpKernel, PowerLawKernel
    from timeline_dashboard import (
        SimulationTimeline,
        plot_timeline_interactive,
        simulate_exp_timeline,
        simulate_powerlaw_timeline,
    )


ensure_bridge_path()

COLORWAY = [
    "#4af699",
    "#f94f6d",
    "#ffd166",
    "#4d9de0",
    "#ff9f1c",
    "#9f7aea",
    "#00c2d1",
    "#f67280",
]
PLOT_BG_HEX = "#0b1220"
GRID_HEX = "#1f2a3c"
FONT_HEX = "#e6edf7"
FONT_FAMILY = '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif'


DEFAULTS: Dict[str, object] = {
    "kernel_choice": "Exponential",
    "mu": 0.2,
    "horizon": 200.0,
    "bins": 100,
    "seed": 12345,
    "exp_alpha": 0.8,
    "exp_beta": 1.2,
    "power_alpha": 0.12,
    "power_c": 0.1,
    "power_gamma": 1.4,
    "mark_choice": "LogNormal",
    "mark_lognorm_mean": 0.0,
    "mark_lognorm_sigma": 0.5,
    "mark_exponential_scale": 1.0,
    "mark_value": 1.0,
    "compare_enabled": False,
    "compare_kernel": "Power-law",
    "compare_exp_alpha": 0.6,
    "compare_exp_beta": 0.9,
    "compare_power_alpha": 0.18,
    "compare_power_c": 0.08,
    "compare_power_gamma": 1.6,
    "candle_seconds": 10,
    "price_scale": 0.25,
    "base_price": 100.0,
}


PRESET_DESCRIPTIONS: Dict[str, str] = {
    "Calm market": (
        "Low background activity with quickly fading memory — a sleepy tape."
    ),
    "Frenzy": (
        "High base flow and long memory that create persistent bursts of trading."
    ),
    "Flash crash": (
        "Moderate base flow but strong self-excitation for sudden cascades."
    ),
}


PRESET_SCENARIOS: Dict[str, Dict[str, object]] = {
    "Custom": {},
    "Calm market": {
        "kernel_choice": "Exponential",
        "mu": 0.08,
        "horizon": 300.0,
        "bins": 120,
        "exp_alpha": 0.4,
        "exp_beta": 1.5,
        "mark_choice": "Deterministic",
        "mark_value": 0.7,
    },
    "Frenzy": {
        "kernel_choice": "Power-law",
        "mu": 0.35,
        "horizon": 200.0,
        "bins": 100,
        "power_alpha": 0.4,
        "power_c": 0.05,
        "power_gamma": 1.25,
        "mark_choice": "LogNormal",
        "mark_lognorm_mean": 0.3,
        "mark_lognorm_sigma": 0.9,
    },
    "Flash crash": {
        "kernel_choice": "Exponential",
        "mu": 0.15,
        "horizon": 120.0,
        "bins": 80,
        "exp_alpha": 1.3,
        "exp_beta": 0.7,
        "mark_choice": "Exponential",
        "mark_exponential_scale": 1.6,
    },
}


def _init_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _apply_preset(values: Dict[str, object]) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.experimental_rerun()


def _exp_controls(key_prefix: str = "") -> Dict[str, float]:
    alpha = st.sidebar.slider(
        "α — self-excitation strength",
        0.01,
        2.0,
        value=float(
            st.session_state.get(f"{key_prefix}exp_alpha", DEFAULTS["exp_alpha"])
        ),
        step=0.01,
        key=f"{key_prefix}exp_alpha",
        help=(
            "Higher α means each trade boosts the intensity of follow-on trades "
            "more strongly."
        ),
    )
    beta = st.sidebar.slider(
        "β — memory decay",
        0.1,
        5.0,
        value=float(
            st.session_state.get(f"{key_prefix}exp_beta", DEFAULTS["exp_beta"])
        ),
        step=0.01,
        key=f"{key_prefix}exp_beta",
        help=(
            "Controls how quickly the excitement fades. Larger β means the market "
            "cools off faster."
        ),
    )
    return {"alpha": alpha, "beta": beta}


def _powerlaw_controls(key_prefix: str = "") -> Dict[str, float]:
    alpha = st.sidebar.slider(
        "α — clustering strength",
        0.01,
        1.0,
        value=float(
            st.session_state.get(f"{key_prefix}power_alpha", DEFAULTS["power_alpha"])
        ),
        step=0.01,
        key=f"{key_prefix}power_alpha",
        help="Higher α makes bursts of activity more intense.",
    )
    c = st.sidebar.slider(
        "c — short-term cushion",
        0.01,
        1.0,
        value=float(st.session_state.get(f"{key_prefix}power_c", DEFAULTS["power_c"])),
        step=0.01,
        key=f"{key_prefix}power_c",
        help=(
            "Acts like a time offset preventing the intensity from blowing up "
            "immediately after an event."
        ),
    )
    gamma = st.sidebar.slider(
        "γ — memory tail",
        1.01,
        3.5,
        value=float(
            st.session_state.get(f"{key_prefix}power_gamma", DEFAULTS["power_gamma"])
        ),
        step=0.01,
        key=f"{key_prefix}power_gamma",
        help="Lower γ keeps memory longer, creating drawn-out cascades of activity.",
    )
    return {"alpha": alpha, "c": c, "gamma": gamma}


MarkSamplerConfig = Dict[str, float | str]
NUMERIC_TYPES = (int, float, np.floating)


def _mark_sampler_controls() -> MarkSamplerConfig:
    choice = st.sidebar.selectbox(
        "Order size distribution",
        ("LogNormal", "Exponential", "Deterministic"),
        key="mark_choice",
        help="Choose how trade sizes (marks) are drawn when the process jumps.",
    )
    if choice == "LogNormal":
        mean = st.sidebar.slider(
            "Log-mean",
            -1.0,
            1.0,
            value=float(
                st.session_state.get("mark_lognorm_mean", DEFAULTS["mark_lognorm_mean"])
            ),
            step=0.05,
            key="mark_lognorm_mean",
            help="Sets the typical log-size of trades during bursts.",
        )
        sigma = st.sidebar.slider(
            "Log-sigma",
            0.1,
            1.5,
            value=float(
                st.session_state.get(
                    "mark_lognorm_sigma", DEFAULTS["mark_lognorm_sigma"]
                )
            ),
            step=0.05,
            key="mark_lognorm_sigma",
            help="Controls how varied the trade sizes are (higher means more spread).",
        )

        return {"kind": "LogNormal", "mean": mean, "sigma": sigma}

    if choice == "Exponential":
        scale = st.sidebar.slider(
            "Scale",
            0.1,
            5.0,
            value=float(
                st.session_state.get(
                    "mark_exponential_scale", DEFAULTS["mark_exponential_scale"]
                )
            ),
            step=0.1,
            key="mark_exponential_scale",
            help="Average order size. Higher scale = larger trades on average.",
        )

        return {"kind": "Exponential", "scale": scale}

    value = st.sidebar.slider(
        "Fixed value",
        0.01,
        5.0,
        value=float(st.session_state.get("mark_value", DEFAULTS["mark_value"])),
        step=0.01,
        key="mark_value",
        help=(
            "All trades use the same mark size — useful for stress-testing the "
            "timing model alone."
        ),
    )

    return {"kind": "Deterministic", "value": value}


def _build_mark_sampler(config: MarkSamplerConfig):
    kind = config.get("kind")
    if kind == "LogNormal":
        mean = float(config["mean"])
        sigma = float(config["sigma"])

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.lognormal(mean, sigma))

    elif kind == "Exponential":
        scale = float(config["scale"])

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.exponential(scale))

    else:
        value = float(config["value"])

        def sampler(rng: np.random.Generator) -> float:
            return float(value)

    sampler.__doc__ = f"Sampler: {kind}"
    return sampler


def _branching_ratio_message(branching_ratio: float | None) -> Tuple[str, str]:
    if branching_ratio is None or np.isnan(branching_ratio):
        return "info", "Branching ratio N/A"
    if np.isinf(branching_ratio) or branching_ratio >= 1.0:
        return "error", f"Branching ratio ≈ {branching_ratio:.2f} — supercritical"
    if branching_ratio >= 0.85:
        return "warning", f"Branching ratio ≈ {branching_ratio:.2f} — near-critical"
    return (
        "success",
        f"Branching ratio ≈ {branching_ratio:.2f} — comfortably subcritical",
    )


@st.cache_data(show_spinner=False)
def _simulate_hawkes(
    kernel_choice: str,
    mu: float,
    params_tuple: Tuple[Tuple[str, float], ...],
    horizon: float,
    bins: int,
    seed: int,
    mark_config_tuple: Tuple[Tuple[str, float | str], ...],
):
    params = dict(params_tuple)
    mark_config = dict(mark_config_tuple)
    mark_sampler = _build_mark_sampler(mark_config)

    if kernel_choice == "Exponential":
        kernel = ExpKernel(**params)
        timeline = simulate_exp_timeline(mu, kernel, horizon, mark_sampler, seed, bins)
        branching_ratio = kernel.branching_ratio(
            np.mean(timeline.marks) if timeline.marks.size else 0.0
        )
        ratio_value = (
            float(branching_ratio)
            if branching_ratio is not None and not np.isnan(branching_ratio)
            else float("nan")
        )
        summary = {
            "Kernel": kernel_choice,
            "α": params["alpha"],
            "β": params["beta"],
            "Branching ratio": ratio_value,
        }
    else:
        kernel = PowerLawKernel(**params)
        timeline = simulate_powerlaw_timeline(
            mu, kernel, horizon, mark_sampler, seed, bins
        )
        branching_ratio = kernel.branching_ratio(
            np.mean(timeline.marks) if timeline.marks.size else 0.0
        )
        summary = {
            "Kernel": kernel_choice,
            "α": params["alpha"],
            "c": params["c"],
            "γ": params["gamma"],
            "Branching ratio": (
                float(branching_ratio)
                if branching_ratio is not None and not np.isnan(branching_ratio)
                else float("nan")
            ),
        }

    summary.update(
        {
            "Events": int(timeline.times.size),
            "Average mark": float(
                np.mean(timeline.marks) if timeline.marks.size else 0.0
            ),
            "Max intensity": float(
                timeline.intensity_grid.max() if timeline.intensity_grid.size else mu
            ),
        }
    )
    return timeline, summary


def _interpolate_intensity(
    timeline: SimulationTimeline, default_value: float = 0.0
) -> np.ndarray:
    if timeline.intensity_grid.size == 0 or timeline.grid_times.size == 0:
        return np.full(timeline.times.shape, default_value)
    return np.interp(
        timeline.times,
        timeline.grid_times,
        timeline.intensity_grid,
        left=timeline.intensity_grid[0],
        right=timeline.intensity_grid[-1],
    )


def _trades_3d_figure(
    timeline: SimulationTimeline,
    primary_label: str,
    comparison: SimulationTimeline | None = None,
    comparison_label: str | None = None,
) -> go.Figure:
    """Three-dimensional view of the simulated order flow."""

    fig = go.Figure()

    if timeline.times.size:
        primary_intensity = _interpolate_intensity(timeline)
        primary_volume = (
            np.cumsum(timeline.marks)
            if timeline.marks.size
            else np.zeros(timeline.times.shape)
        )
        fig.add_trace(
            go.Scatter3d(
                x=timeline.times,
                y=primary_volume,
                z=primary_intensity,
                mode="markers",
                name=primary_label,
                marker=dict(
                    size=4,
                    color=primary_intensity,
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar=dict(
                        title=dict(
                            text="λ(t)",
                            font=dict(color=FONT_HEX, family=FONT_FAMILY),
                        ),
                        tickfont=dict(color=FONT_HEX, family=FONT_FAMILY),
                        bgcolor="rgba(11,18,32,0.8)",
                    ),
                ),
                hovertemplate=(
                    "t = %{x:.2f}s<br>Cumulative volume = %{y:.3f}"
                    "<br>λ(t) = %{z:.3f}<extra></extra>"
                ),
            )
        )

    if comparison is not None and comparison.times.size:
        comparison_intensity = _interpolate_intensity(comparison)
        comparison_volume = (
            np.cumsum(comparison.marks)
            if comparison.marks.size
            else np.zeros(comparison.times.shape)
        )
        fig.add_trace(
            go.Scatter3d(
                x=comparison.times,
                y=comparison_volume,
                z=comparison_intensity,
                mode="markers",
                name=comparison_label or "Comparison",
                marker=dict(
                    size=4,
                    color=OVERLAY_INTENSITY_COLOR,
                    opacity=0.6,
                ),
                hovertemplate=(
                    "t = %{x:.2f}s<br>Cumulative volume = %{y:.3f}"
                    "<br>λ(t) = %{z:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="3D order flow projection",
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="Time (s)",
                    font=dict(color=FONT_HEX, family=FONT_FAMILY),
                ),
                backgroundcolor="rgba(17,26,44,0.6)",
                gridcolor=GRID_HEX,
                zerolinecolor=GRID_HEX,
                tickfont=dict(color=FONT_HEX, family=FONT_FAMILY),
            ),
            yaxis=dict(
                title=dict(
                    text="Cumulative volume",
                    font=dict(color=FONT_HEX, family=FONT_FAMILY),
                ),
                backgroundcolor="rgba(17,26,44,0.6)",
                gridcolor=GRID_HEX,
                zerolinecolor=GRID_HEX,
                tickfont=dict(color=FONT_HEX, family=FONT_FAMILY),
            ),
            zaxis=dict(
                title=dict(
                    text="Intensity λ(t)",
                    font=dict(color=FONT_HEX, family=FONT_FAMILY),
                ),
                backgroundcolor="rgba(17,26,44,0.6)",
                gridcolor=GRID_HEX,
                zerolinecolor=GRID_HEX,
                tickfont=dict(color=FONT_HEX, family=FONT_FAMILY),
            ),
        ),
        paper_bgcolor=PLOT_BG_HEX,
        font=dict(color=FONT_HEX, family=FONT_FAMILY),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(11, 18, 32, 0.85)",
            bordercolor=GRID_HEX,
            borderwidth=1,
        ),
    )
    if not fig.data:
        fig.add_annotation(
            text="No events to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=FONT_HEX, family=FONT_FAMILY, size=16),
        )
    return fig


def main() -> None:
    _init_defaults()

    st.set_page_config(page_title="Hawkes Simulator", layout="wide")
    st.title("High-Frequency Order Flow Simulator")
    st.caption("Visualizing clustered order arrivals with Hawkes processes")
    st.markdown(
        (
            'This app simulates how trades in a market can "snowball", where one '
            "trade nudges the next."
        )
    )

    with st.expander("Learn more"):
        st.write("""
            Hawkes processes model event clustering in finance, seismology, and even
            social media. In markets they describe bursts of trading activity: a burst
            of orders today makes another burst more likely a moment later. Experiment
            with the base activity (μ), how long memory lasts, and the distribution of
            order sizes to see calm periods, frenzies, or near-critical cascades
            emerge. No prior microstructure knowledge is needed—just move the sliders
            and watch how the timeline reacts.
            """)

    st.sidebar.header("Market regimes")

    preset_name = st.sidebar.selectbox(
        "Choose a regime",
        list(PRESET_SCENARIOS.keys()),
        key="preset_choice",
        help=(
            "Pick a canned market regime to auto-fill parameters or stay on Custom "
            "to explore freely."
        ),
    )
    if preset_name != "Custom" and st.sidebar.button(
        f"Load '{preset_name}'",
        help="Apply the preset values and rerun the simulation.",
    ):
        _apply_preset(PRESET_SCENARIOS[preset_name])
    if preset_name in PRESET_DESCRIPTIONS:
        st.sidebar.caption(PRESET_DESCRIPTIONS[preset_name])

    kernel_choice = st.sidebar.selectbox(
        "Kernel",
        ("Exponential", "Power-law"),
        key="kernel_choice",
        help=(
            "Exponential forgets quickly; power-law keeps long memory and heavier "
            "clustering."
        ),
    )
    mu = st.sidebar.slider(
        "Base intensity μ",
        0.01,
        2.0,
        value=float(st.session_state.get("mu", DEFAULTS["mu"])),
        step=0.01,
        key="mu",
        help="Background arrival rate when nothing exciting is happening.",
    )
    horizon = st.sidebar.slider(
        "Horizon T",
        10.0,
        1000.0,
        value=float(st.session_state.get("horizon", DEFAULTS["horizon"])),
        step=10.0,
        key="horizon",
        help="How long to simulate the process (in seconds).",
    )
    bins = st.sidebar.slider(
        "Timeline bins",
        20,
        200,
        value=int(st.session_state.get("bins", DEFAULTS["bins"])),
        step=5,
        key="bins",
        help="Controls the resolution of the counts plot (more bins = finer detail).",
    )
    seed = st.sidebar.number_input(
        "Seed",
        value=int(st.session_state.get("seed", DEFAULTS["seed"])),
        step=1,
        key="seed",
        help=(
            "Fix the random seed so runs are repeatable. Change it for a fresh "
            "sample path."
        ),
    )
    mark_config = _mark_sampler_controls()

    if kernel_choice == "Exponential":
        params = _exp_controls()
        params_tuple = tuple(
            sorted((key, float(value)) for key, value in params.items())
        )
    else:
        params = _powerlaw_controls()
        params_tuple = tuple(
            sorted((key, float(value)) for key, value in params.items())
        )

    mark_config_tuple = tuple(sorted(mark_config.items()))
    timeline, summary = _simulate_hawkes(
        kernel_choice,
        float(mu),
        params_tuple,
        float(horizon),
        int(bins),
        int(seed),
        mark_config_tuple,
    )

    comparison_timeline = None
    comparison_label = None
    comparison_summary = None
    with st.sidebar.expander(
        "Comparison overlay",
        expanded=bool(st.session_state.get("compare_enabled", False)),
    ):
        compare_enabled = st.checkbox(
            "Overlay a second kernel",
            value=bool(st.session_state.get("compare_enabled", False)),
            key="compare_enabled",
            help=(
                "Run a second set of parameters and plot both on the same charts to "
                "compare regimes."
            ),
        )
        if compare_enabled:
            compare_kernel_choice = st.selectbox(
                "Comparison kernel",
                ("Exponential", "Power-law"),
                key="compare_kernel",
                help="Choose the structure for the comparison run.",
            )
            if compare_kernel_choice == "Exponential":
                compare_params = _exp_controls("compare_")
                compare_params_tuple = tuple(
                    sorted((key, float(value)) for key, value in compare_params.items())
                )
                comparison_timeline, comparison_summary = _simulate_hawkes(
                    compare_kernel_choice,
                    float(mu),
                    compare_params_tuple,
                    float(horizon),
                    int(bins),
                    int(seed) + 1,
                    mark_config_tuple,
                )
                comparison_label = f"{compare_kernel_choice} (overlay)"
            else:
                compare_params = _powerlaw_controls("compare_")
                compare_params_tuple = tuple(
                    sorted((key, float(value)) for key, value in compare_params.items())
                )
                comparison_timeline, comparison_summary = _simulate_hawkes(
                    compare_kernel_choice,
                    float(mu),
                    compare_params_tuple,
                    float(horizon),
                    int(bins),
                    int(seed) + 1,
                    mark_config_tuple,
                )
                comparison_label = f"{compare_kernel_choice} (overlay)"

    col_plot, col_data = st.columns((2.2, 1))
    with col_plot:
        timeline_tab, view3d_tab = st.tabs(["Timeline", "3D flow"])

        with timeline_tab:
            fig = plot_timeline_interactive(
                timeline,
                title=f"{kernel_choice} Hawkes Simulation",
                comparison=comparison_timeline,
                labels=(kernel_choice, comparison_label or "Comparison"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with view3d_tab:
            fig3d = _trades_3d_figure(
                timeline,
                kernel_choice,
                comparison=comparison_timeline,
                comparison_label=comparison_label,
            )
            st.plotly_chart(fig3d, use_container_width=True)

    with col_data:
        st.subheader("Summary")
        scenarios = [
            (kernel_choice, "Baseline", timeline, summary),
        ]
        if comparison_timeline is not None and comparison_summary is not None:
            scenarios.append(
                (
                    comparison_label or "Comparison",
                    "Overlay",
                    comparison_timeline,
                    comparison_summary,
                )
            )

        metric_cols = st.columns(len(scenarios))
        base_summary = summary
        for idx, (label, _subtitle, _timeline_obj, stats) in enumerate(scenarios):
            with metric_cols[idx]:
                st.markdown(f"**{label}**")
                delta_events = None
                delta_avg_mark = None
                delta_intensity = None
                delta_branching = None
                if idx > 0:
                    delta_events = stats["Events"] - base_summary["Events"]
                    delta_avg_mark = (
                        stats["Average mark"] - base_summary["Average mark"]
                    )
                    delta_intensity = (
                        stats["Max intensity"] - base_summary["Max intensity"]
                    )
                    if (
                        isinstance(stats["Branching ratio"], NUMERIC_TYPES)
                        and isinstance(base_summary["Branching ratio"], NUMERIC_TYPES)
                        and not np.isnan(stats["Branching ratio"])
                        and not np.isnan(base_summary["Branching ratio"])
                    ):
                        delta_branching = (
                            stats["Branching ratio"] - base_summary["Branching ratio"]
                        )
                st.metric(
                    "Events",
                    stats["Events"],
                    f"{delta_events:+d}" if delta_events is not None else None,
                )
                st.metric(
                    "Average mark",
                    f"{stats['Average mark']:.3f}",
                    f"{delta_avg_mark:+.3f}" if delta_avg_mark is not None else None,
                )
                st.metric(
                    "Max intensity",
                    f"{stats['Max intensity']:.3f}",
                    f"{delta_intensity:+.3f}" if delta_intensity is not None else None,
                )
                severity, msg = _branching_ratio_message(stats["Branching ratio"])
                show_message = msg
                if idx == 0:
                    if severity == "error":
                        st.error(show_message)
                    elif severity == "warning":
                        st.warning(show_message)
                    elif severity == "success":
                        st.success(show_message)
                    else:
                        st.info(show_message)
                else:
                    if delta_branching is not None:
                        show_message = f"{msg} (Δ {delta_branching:+.2f})"
                    if severity == "error":
                        st.error(show_message)
                    elif severity == "warning":
                        st.warning(show_message)
                    elif severity == "success":
                        st.success(show_message)
                    else:
                        st.info(show_message)

        summary_table = pd.DataFrame(
            [
                {
                    "Scenario": label,
                    **{k: v for k, v in stats.items() if k != "Branching ratio"},
                    "Branching ratio": (
                        stats["Branching ratio"]
                        if isinstance(stats["Branching ratio"], NUMERIC_TYPES)
                        else np.nan
                    ),
                }
                for label, _, _, stats in scenarios
            ]
        ).set_index("Scenario")
        st.dataframe(summary_table, use_container_width=True)

        if timeline.marks.size:
            hist_fig = go.Figure()
            base_color = COLORWAY[0]
            hist_fig.add_trace(
                go.Histogram(
                    x=timeline.marks,
                    name=kernel_choice,
                    nbinsx=min(30, max(5, int(len(timeline.marks) / 5))),
                    opacity=0.75,
                    marker_color=base_color,
                )
            )
            if comparison_timeline is not None and comparison_timeline.marks.size:
                comparison_color = COLORWAY[1]
                hist_fig.add_trace(
                    go.Histogram(
                        x=comparison_timeline.marks,
                        name=comparison_label or "Comparison",
                        nbinsx=min(30, max(5, int(len(comparison_timeline.marks) / 5))),
                        opacity=0.6,
                        marker_color=comparison_color,
                    )
                )
                hist_fig.update_layout(barmode="overlay")
            hist_fig.update_layout(
                title="Order size distribution",
                xaxis_title="Mark value",
                yaxis_title="Frequency",
                template=None,
                colorway=COLORWAY,
                plot_bgcolor=PLOT_BG_HEX,
                paper_bgcolor=PLOT_BG_HEX,
                font=dict(color=FONT_HEX),
                bargap=0.15,
            )
            hist_fig.update_xaxes(gridcolor=GRID_HEX, zerolinecolor=GRID_HEX)
            hist_fig.update_yaxes(gridcolor=GRID_HEX, zerolinecolor=GRID_HEX)
            st.plotly_chart(hist_fig, use_container_width=True)

        st.download_button(
            "Export simulated order flow (CSV)",
            data="time,mark\n"
            + "\n".join(
                f"{t:.6f},{m:.6f}" for t, m in zip(timeline.times, timeline.marks)
            ),
            file_name="hawkes_events.csv",
            mime="text/csv",
        )

    notebooks_dir = Path(__file__).resolve().parent.parent / "docs" / "notebooks"
    if notebooks_dir.exists():
        st.sidebar.markdown("---")
        st.sidebar.subheader("Go deeper")
        st.sidebar.info(
            (
                "See how we calibrate Hawkes models on real Binance order flow and "
                "synthetic data."
            )
        )
        for notebook_path in notebooks_dir.glob("*.ipynb"):
            notebook_bytes = notebook_path.read_bytes()
            st.sidebar.download_button(
                label=(
                    "Download "
                    f"{notebook_path.stem.replace('_', ' ').title()} notebook"
                ),
                data=notebook_bytes,
                file_name=notebook_path.name,
                mime="application/x-ipynb+json",
            )


if __name__ == "__main__":
    main()
