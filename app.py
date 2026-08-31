"""
================================================================================
 Superpave Gyratory Compaction Curve Predictor
 CatBoost multi-output model  ->  Slope (a) and Intercept (b) of
        AV(N_g) = a * ln(N_g) + b
================================================================================
 Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Compaction Curve Predictor",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = Path(__file__).parent
MODEL_PKL = HERE / "compaction_model.pkl"
META_JSON = HERE / "model_metadata.json"

PALETTE = dict(
    ink="#0F2A43", blue="#1F4E79", red="#B03A2E", green="#2E7D5B",
    amber="#B7791F", light="#8FA9C4", paper="#F7F9FC", line="#DDE5EE",
)

# ══════════════════════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

  .hero {{
      background: linear-gradient(125deg, {PALETTE['ink']} 0%, {PALETTE['blue']} 55%, #2E5F8A 100%);
      border-radius: 16px; padding: 1.7rem 2.1rem; margin-bottom: 1.4rem;
      color: #fff; box-shadow: 0 10px 30px rgba(15,42,67,.22);
      position: relative; overflow: hidden;
  }}
  .hero:after {{
      content:""; position:absolute; right:-70px; top:-70px;
      width:260px; height:260px; border-radius:50%;
      background: radial-gradient(circle, rgba(255,255,255,.13) 0%, transparent 70%);
  }}
  .hero h1 {{
      font-family:'Source Serif 4', Georgia, serif;
      font-size: 2.0rem; font-weight: 600; margin: 0 0 .35rem 0; letter-spacing:-.4px;
  }}
  .hero p  {{ margin:0; opacity:.90; font-size:.97rem; line-height:1.55; max-width: 62rem;}}
  .hero .eq {{
      display:inline-block; margin-top:.85rem; padding:.42rem .9rem;
      background: rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.24);
      border-radius: 8px; font-family:'Source Serif 4', Georgia, serif; font-size:1.02rem;
  }}

  .card {{
      background:#fff; border:1px solid {PALETTE['line']}; border-radius:14px;
      padding:1.15rem 1.35rem; box-shadow:0 2px 10px rgba(15,42,67,.06); height:100%;
  }}
  .card .lab {{ font-size:.74rem; letter-spacing:.10em; text-transform:uppercase;
                color:#6B7C93; font-weight:600; margin-bottom:.30rem; }}
  .card .val {{ font-size:2.35rem; font-weight:700; line-height:1.05; letter-spacing:-1px; }}
  .card .sub {{ font-size:.82rem; color:#7A8A9E; margin-top:.32rem; }}

  .chip {{ display:inline-block; padding:.22rem .68rem; border-radius:999px;
           font-size:.76rem; font-weight:600; margin:.16rem .28rem .16rem 0; }}
  .ok   {{ background:#E7F4EE; color:#1E6B4C; border:1px solid #BFE3D2; }}
  .warn {{ background:#FDF3E0; color:#8A5A00; border:1px solid #F2DDB4; }}
  .bad  {{ background:#FBEAE7; color:#8E2D20; border:1px solid #F3CFC9; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid {PALETTE['line']}; }}
  .stTabs [data-baseweb="tab"] {{
      height:44px; padding:0 1.05rem; border-radius:9px 9px 0 0;
      font-weight:600; font-size:.92rem; color:#5A6B80;
  }}
  .stTabs [aria-selected="true"] {{ background:{PALETTE['paper']}; color:{PALETTE['blue']}; }}

  section[data-testid="stSidebar"] {{ background:{PALETTE['paper']}; border-right:1px solid {PALETTE['line']}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top:1.2rem; }}

  .stButton>button {{
      border-radius:9px; font-weight:600; border:1px solid {PALETTE['line']};
      transition:all .15s ease;
  }}
  .stButton>button:hover {{ border-color:{PALETTE['blue']}; color:{PALETTE['blue']}; }}

  .foot {{ color:#8A9AAD; font-size:.80rem; text-align:center;
           border-top:1px solid {PALETTE['line']}; padding-top:1rem; margin-top:2.2rem; }}
  div[data-testid="stMetricValue"] {{ font-size:1.5rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL + METADATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading the trained model…")
def load_model():
    with open(MODEL_PKL, "rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def load_meta():
    with open(META_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


try:
    model = load_model()
    META = load_meta()
except Exception as exc:                                        # noqa: BLE001
    st.error(f"Could not load the model artefacts: {exc}")
    st.stop()

FEATURES: list[str] = META["features"]          # exact training column order
LABELS: dict = META["feature_labels"]
HELP: dict = META["feature_help"]
STATS: dict = META["feature_stats"]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict(row: dict) -> tuple[float, float]:
    """row -> (slope, intercept). Column order is taken from the metadata."""
    df = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    a, b = model.predict(df)[0]
    return float(a), float(b)


def av_curve(a: float, b: float, n_max: int = 200) -> tuple[np.ndarray, np.ndarray]:
    ng = np.arange(1, n_max + 1)
    return ng, a * np.log(ng) + b


def n_at_av(a: float, b: float, av: float = 4.0) -> float:
    if a == 0:
        return float("nan")
    return float(np.exp((av - b) / a))


def extrapolation_flags(row: dict) -> dict:
    out = {}
    for f in FEATURES:
        s = STATS[f]
        if row[f] < s["min"] or row[f] > s["max"]:
            out[f] = "outside"
        elif row[f] < s["p5"] or row[f] > s["p95"]:
            out[f] = "edge"
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>Superpave Gyratory Compaction Curve Predictor</h1>
  <p>A CatBoost multi-output model that predicts both coefficients of the logarithmic
     densification law from binder properties, compaction temperature and aggregate
     gradation descriptors - then reconstructs the full air-void curve.</p>
  <div class="eq">AV(N<sub>g</sub>) &nbsp;=&nbsp; a &middot; ln(N<sub>g</sub>) &nbsp;+&nbsp; b</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Mixture inputs")

    preset = st.radio(
    "Starting Point",
    [
        "Typical baseline (median)",
        "Low-input scenario (P5)",
        "High-input scenario (P95)"
    ],
    help=(
        "Automatically initializes all mixture-input controls using "
        "the training-data distribution. Each input can then be adjusted."
    )
)

    def default_for(f: str) -> float:
        s = STATS[f]
        return {"Typical baseline (median)": s["median"],
                "Low-input scenario (P5)": s["p5"],
                "High-input scenario (P95)": s["p95"]}[preset]

    st.divider()
    values: dict = {}
    for f in FEATURES:
        s = STATS[f]
        label = f"{LABELS[f]}"
        if s["levels"]:                                   # discrete feature
            lv = s["levels"]
            d = min(lv, key=lambda v: abs(v - default_for(f)))
            values[f] = st.select_slider(
                label, options=lv, value=d, help=HELP.get(f, ""),
                key=f"in_{f}", format_func=lambda v: f"{v:,.0f}")
        else:
            span = s["max"] - s["min"]
            step = float(f"{span/200:.2g}") or 0.01
            values[f] = st.slider(
                label,
                min_value=float(s["min"] - 0.10 * span),
                max_value=float(s["max"] + 0.10 * span),
                value=float(default_for(f)), step=step,
                help=f"{HELP.get(f,'')}  ·  training range "
                     f"[{s['min']:,.3g}, {s['max']:,.3g}]",
                key=f"in_{f}")

    st.divider()
    st.markdown("##### Gyration criteria")
    c1, c2, c3 = st.columns(3)
    N_ini = c1.number_input("N$_{ini}$", 1, 50, 7, key="nini")
    N_des = c2.number_input("N$_{des}$", 10, 250, 75, key="ndes")
    N_max = c3.number_input("N$_{max}$", 20, 500, 115, key="nmax")

    st.caption("Superpave criteria: %G$_{mm}$ ≤ 89 at N$_{ini}$, "
               "= 96.0 at N$_{des}$, ≤ 98 at N$_{max}$.")

# prediction
slope, intercept = predict(values)
flags = extrapolation_flags(values)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_pred, tab_sens, tab_batch, tab_card = st.tabs(
    ["  Prediction  ", "  Sensitivity  ", "  Batch prediction  ", "  Model card  "])

# ─────────────────────────────────────────────────────── PREDICTION ───────────
with tab_pred:
    ng, av = av_curve(slope, intercept, 200)
    gmm = 100.0 - av
    av_ini, av_des, av_mx = [slope * np.log(n) + intercept for n in (N_ini, N_des, N_max)]
    n4 = n_at_av(slope, intercept, 4.0)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""<div class="card"><div class="lab">Slope&nbsp;·&nbsp;a</div>
        <div class="val" style="color:{PALETTE['blue']}">{slope:.4f}</div>
        <div class="sub">densification rate per ln-gyration</div></div>""",
        unsafe_allow_html=True)
    k2.markdown(f"""<div class="card"><div class="lab">Intercept&nbsp;·&nbsp;b</div>
        <div class="val" style="color:{PALETTE['red']}">{intercept:.4f}</div>
        <div class="sub">air voids at N<sub>g</sub> = 1 (%)</div></div>""",
        unsafe_allow_html=True)
    k3.markdown(f"""<div class="card"><div class="lab">AV at N<sub>des</sub></div>
        <div class="val" style="color:{PALETTE['green']}">{av_des:.2f}<span
        style="font-size:1.1rem"> %</span></div>
        <div class="sub">%G<sub>mm</sub> = {100-av_des:.2f}</div></div>""",
        unsafe_allow_html=True)
    n4_txt = f"{n4:,.0f}" if np.isfinite(n4) and n4 < 1e4 else "-"
    k4.markdown(f"""<div class="card"><div class="lab">N<sub>g</sub> to reach 4 % AV</div>
        <div class="val" style="color:{PALETTE['amber']}">{n4_txt}</div>
        <div class="sub">gyrations</div></div>""", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin:1.1rem 0 .4rem 0;font-family:Source Serif 4,Georgia,serif;"
        f"font-size:1.20rem;color:{PALETTE['ink']}'>"
        f"<b>AV(N<sub>g</sub>) = {slope:.4f} · ln(N<sub>g</sub>) + {intercept:.4f}</b></div>",
        unsafe_allow_html=True)

    # criteria chips
    chips = []
    for lbl, val, rule, ok in [
        (f"%Gmm @ N_ini={N_ini}", 100 - av_ini, "≤ 89.0", (100 - av_ini) <= 89.0),
        (f"%Gmm @ N_des={N_des}", 100 - av_des, "≈ 96.0", abs((100 - av_des) - 96.0) <= 1.0),
        (f"%Gmm @ N_max={N_max}", 100 - av_mx, "≤ 98.0", (100 - av_mx) <= 98.0),
    ]:
        chips.append(f"<span class='chip {'ok' if ok else 'bad'}'>{lbl}: "
                     f"{val:.2f} &nbsp;({rule})</span>")
    if flags:
        names = ", ".join(LABELS[f] for f in flags)
        chips.append(f"<span class='chip warn'>Extrapolation: {names}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)

    # ── the model figure ──────────────────────────────────────────────────────
        # ── The model figure ──────────────────────────────────────────────────────

    # Ensure that the predicted curve includes Nmax.
    # The axis will normally extend from 1 to 200 gyrations,
    # but it will extend farther if the user selects Nmax > 200.
    plot_n_max = max(200, int(N_max))

    ng, av = av_curve(slope, intercept, plot_n_max)
    gmm = 100.0 - av

    av_ini, av_des, av_mx = [
        slope * np.log(n) + intercept
        for n in (N_ini, N_des, N_max)
    ]

    # Ranges used to draw the vertical design-gyration lines.
    av_y_min = float(
        min(np.nanmin(av - 0.30), av_ini, av_des, av_mx, 4.0)
    )
    av_y_max = float(
        max(np.nanmax(av + 0.30), av_ini, av_des, av_mx, 4.0)
    )

    gmm_y_min = float(min(np.nanmin(gmm), 89.0))
    gmm_y_max = float(max(np.nanmax(gmm), 98.0))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            "Predicted air-void densification curve",
            "Degree of compaction, %G<sub>mm</sub>"
        )
    )

    # ── Top panel: predicted air-void curve ───────────────────────────────────

    fig.add_trace(
        go.Scatter(
            x=ng,
            y=av,
            mode="lines",
            name="Predicted AV",
            line=dict(
                color=PALETTE["blue"],
                width=3.4
            ),
            hovertemplate=(
                "N<sub>g</sub> = %{x}<br>"
                "AV = %{y:.2f} %"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # LOOCV uncertainty band
    fig.add_trace(
        go.Scatter(
            x=np.r_[ng, ng[::-1]],
            y=np.r_[av + 0.30, (av - 0.30)[::-1]],
            fill="toself",
            fillcolor="rgba(31,78,121,.13)",
            line=dict(width=0),
            name="≈ LOOCV uncertainty (±0.3 %)",
            hoverinfo="skip"
        ),
        row=1,
        col=1
    )

    # Four-percent air-void target
    fig.add_hline(
        y=4.0,
        line=dict(
            color=PALETTE["red"],
            dash="dot",
            width=2
        ),
        annotation_text="4 % design target",
        annotation_position="top right",
        row=1,
        col=1
    )

    # ── Bottom panel: degree of compaction ────────────────────────────────────

    fig.add_trace(
        go.Scatter(
            x=ng,
            y=gmm,
            mode="lines",
            name="%G<sub>mm</sub>",
            line=dict(
                color=PALETTE["green"],
                width=3.0
            ),
            showlegend=False,
            hovertemplate=(
                "N<sub>g</sub> = %{x}<br>"
                "%G<sub>mm</sub> = %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=2,
        col=1
    )

    # ── Design-gyration vertical lines ────────────────────────────────────────
    #
    # These are drawn as scatter traces instead of fig.add_vline().
    # Scatter traces use the actual N values correctly on a logarithmic axis.

    design_lines = [
        (N_ini, "N<sub>ini</sub>", PALETTE["amber"]),
        (N_des, "N<sub>des</sub>", PALETTE["green"]),
        (N_max, "N<sub>max</sub>", PALETTE["light"])
    ]

    for n, label, color in design_lines:

        # Vertical line and label in the air-void panel
        fig.add_trace(
            go.Scatter(
                x=[n, n],
                y=[av_y_min, av_y_max],
                mode="lines+text",
                text=["", label],
                textposition="top center",
                textfont=dict(
                    size=11,
                    color=PALETTE["ink"]
                ),
                line=dict(
                    color=color,
                    dash="dash",
                    width=1.6
                ),
                showlegend=False,
                hoverinfo="skip",
                cliponaxis=False
            ),
            row=1,
            col=1
        )

        # Vertical line in the degree-of-compaction panel
        fig.add_trace(
            go.Scatter(
                x=[n, n],
                y=[gmm_y_min, gmm_y_max],
                mode="lines",
                line=dict(
                    color=color,
                    dash="dash",
                    width=1.6
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=2,
            col=1
        )

    # Design-gyration markers
    fig.add_trace(
        go.Scatter(
            x=[N_ini, N_des, N_max],
            y=[av_ini, av_des, av_mx],
            mode="markers+text",
            text=[
                f"{av_ini:.2f}%",
                f"{av_des:.2f}%",
                f"{av_mx:.2f}%"
            ],
            textposition="bottom right",
            textfont=dict(size=11),
            marker=dict(
                size=11,
                color=PALETTE["red"],
                line=dict(
                    color="white",
                    width=2
                )
            ),
            name="Design gyrations",
            hovertemplate=(
                "N = %{x}<br>"
                "AV = %{y:.2f} %"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # ── Superpave degree-of-compaction limits ─────────────────────────────────

    compaction_limits = [
        (89.0, "89 (N_ini limit)"),
        (96.0, "96 (N_des)"),
        (98.0, "98 (N_max limit)")
    ]

    for y_value, label in compaction_limits:
        fig.add_hline(
            y=y_value,
            line=dict(
                color="#9AA9BB",
                dash="dot",
                width=1.3
            ),
            annotation_text=label,
            annotation_position="bottom left",
            row=2,
            col=1
        )

    # ── Correct logarithmic x-axis ─────────────────────────────────────────────

    # Plotly uses log10 values when specifying a log-axis range.
    x_axis_range = [
        0,
        np.log10(plot_n_max) + 0.03
    ]

    possible_ticks = [
        1, 2, 5, 10, 20, 50, 100, 200, 500
    ]

    tick_values = [
        tick for tick in possible_ticks
        if tick <= plot_n_max
    ]

    tick_labels = [
        f"{tick:,}" for tick in tick_values
    ]

    # Top x-axis
    fig.update_xaxes(
        type="log",
        range=x_axis_range,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        gridcolor=PALETTE["line"],
        row=1,
        col=1
    )

    # Bottom x-axis
    fig.update_xaxes(
        type="log",
        range=x_axis_range,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        title_text=(
            "Number of gyrations, "
            "N<sub>g</sub> (log scale)"
        ),
        gridcolor=PALETTE["line"],
        row=2,
        col=1
    )

    # ── Y-axis formatting ──────────────────────────────────────────────────────

    fig.update_yaxes(
        title_text="Air voids, AV (%)",
        gridcolor=PALETTE["line"],
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="%G<sub>mm</sub>",
        gridcolor=PALETTE["line"],
        row=2,
        col=1
    )

    # ── Figure layout ──────────────────────────────────────────────────────────

    fig.update_layout(
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Inter, sans-serif",
            size=13,
            color=PALETTE["ink"]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(
            l=70,
            r=30,
            t=70,
            b=60
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ── Download buttons ───────────────────────────────────────────────────────

    curve_df = pd.DataFrame({
        "N_gyrations": ng,
        "AV_percent": av,
        "pct_Gmm": gmm
    })

    d1, d2 = st.columns(2)

    d1.download_button(
        "Download predicted curve (CSV)",
        curve_df.to_csv(index=False).encode(),
        "predicted_compaction_curve.csv",
        "text/csv",
        use_container_width=True
    )

    inp = pd.DataFrame([values]).assign(
        Predicted_Slope=slope,
        Predicted_Intercept=intercept
    )

    d2.download_button(
        "Download inputs + prediction (CSV)",
        inp.to_csv(index=False).encode(),
        "prediction_record.csv",
        "text/csv",
        use_container_width=True
    )

    # ── Input summary ──────────────────────────────────────────────────────────

    with st.expander("Input summary"):

        show = pd.DataFrame({
            "Feature": [
                LABELS[f] for f in FEATURES
            ],
            "Value": [
                values[f] for f in FEATURES
            ],
            "Training min": [
                STATS[f]["min"] for f in FEATURES
            ],
            "Training median": [
                STATS[f]["median"] for f in FEATURES
            ],
            "Training max": [
                STATS[f]["max"] for f in FEATURES
            ],
            "Status": [
                "outside training range"
                if flags.get(f) == "outside"
                else (
                    "near the edge"
                    if flags.get(f) == "edge"
                    else "within range"
                )
                for f in FEATURES
            ]
        })

        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True
        )
    
    
    
    
    # fig = make_subplots(
    #     rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
    #     row_heights=[0.62, 0.38],
    #     subplot_titles=("Predicted air-void densification curve",
    #                     "Degree of compaction, %G<sub>mm</sub>"))

    # fig.add_trace(go.Scatter(
    #     x=ng, y=av, mode="lines", name="Predicted AV",
    #     line=dict(color=PALETTE["blue"], width=3.4),
    #     hovertemplate="N<sub>g</sub> = %{x}<br>AV = %{y:.2f} %<extra></extra>"),
    #     row=1, col=1)
    # fig.add_trace(go.Scatter(
    #     x=np.r_[ng, ng[::-1]], y=np.r_[av + 0.30, (av - 0.30)[::-1]],
    #     fill="toself", fillcolor="rgba(31,78,121,.13)", line=dict(width=0),
    #     name="≈ LOOCV uncertainty (±0.3 %)", hoverinfo="skip"), row=1, col=1)
    # fig.add_hline(y=4.0, line=dict(color=PALETTE["red"], dash="dot", width=2),
    #               annotation_text="4 % design target", annotation_position="top right",
    #               row=1, col=1)

    # for n, lbl, col in [(N_ini, "N<sub>ini</sub>", PALETTE["amber"]),
    #                     (N_des, "N<sub>des</sub>", PALETTE["green"]),
    #                     (N_max, "N<sub>max</sub>", PALETTE["light"])]:
    #     fig.add_vline(x=n, line=dict(color=col, dash="dash", width=1.6),
    #                   annotation_text=lbl, annotation_position="top", row=1, col=1)
    #     fig.add_vline(x=n, line=dict(color=col, dash="dash", width=1.6), row=2, col=1)
    # fig.add_trace(go.Scatter(
    #     x=[N_ini, N_des, N_max], y=[av_ini, av_des, av_mx], mode="markers+text",
    #     text=[f"{v:.2f}%" for v in (av_ini, av_des, av_mx)],
    #     textposition="bottom right", textfont=dict(size=11),
    #     marker=dict(size=11, color=PALETTE["red"], line=dict(color="white", width=2)),
    #     name="Design gyrations", hovertemplate="N = %{x}<br>AV = %{y:.2f} %<extra></extra>"),
    #     row=1, col=1)

    # fig.add_trace(go.Scatter(
    #     x=ng, y=gmm, mode="lines", name="%G<sub>mm</sub>",
    #     line=dict(color=PALETTE["green"], width=3.0), showlegend=False,
    #     hovertemplate="N<sub>g</sub> = %{x}<br>%G<sub>mm</sub> = %{y:.2f}<extra></extra>"),
    #     row=2, col=1)
    # for yv, lbl in [(89.0, "89 (N_ini limit)"), (96.0, "96 (N_des)"), (98.0, "98 (N_max limit)")]:
    #     fig.add_hline(y=yv, line=dict(color="#9AA9BB", dash="dot", width=1.3),
    #                   annotation_text=lbl, annotation_position="bottom left", row=2, col=1)

    # fig.update_xaxes(type="log", title_text="Number of gyrations, N<sub>g</sub>  (log scale)",
    #                  row=2, col=1, gridcolor=PALETTE["line"])
    # fig.update_xaxes(type="log", row=1, col=1, gridcolor=PALETTE["line"])
    # fig.update_yaxes(title_text="Air voids, AV (%)", row=1, col=1, gridcolor=PALETTE["line"])
    # fig.update_yaxes(title_text="%G<sub>mm</sub>", row=2, col=1, gridcolor=PALETTE["line"])
    # fig.update_layout(
    #     height=680, plot_bgcolor="white", paper_bgcolor="white",
    #     font=dict(family="Inter, sans-serif", size=13, color=PALETTE["ink"]),
    #     legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
    #                 bgcolor="rgba(0,0,0,0)"),
    #     margin=dict(l=70, r=30, t=70, b=60), hovermode="x unified")
    # st.plotly_chart(fig, use_container_width=True)

    # curve_df = pd.DataFrame({"N_gyrations": ng, "AV_percent": av, "pct_Gmm": gmm})
    # d1, d2 = st.columns(2)
    # d1.download_button("Download predicted curve (CSV)",
    #                    curve_df.to_csv(index=False).encode(),
    #                    "predicted_compaction_curve.csv", "text/csv",
    #                    use_container_width=True)
    # inp = pd.DataFrame([values]).assign(Predicted_Slope=slope, Predicted_Intercept=intercept)
    # d2.download_button("Download inputs + prediction (CSV)",
    #                    inp.to_csv(index=False).encode(),
    #                    "prediction_record.csv", "text/csv", use_container_width=True)

    # with st.expander("Input summary"):
    #     show = pd.DataFrame({
    #         "Feature": [LABELS[f] for f in FEATURES],
    #         "Value": [values[f] for f in FEATURES],
    #         "Training min": [STATS[f]["min"] for f in FEATURES],
    #         "Training median": [STATS[f]["median"] for f in FEATURES],
    #         "Training max": [STATS[f]["max"] for f in FEATURES],
    #         "Status": ["outside training range" if flags.get(f) == "outside"
    #                    else ("near the edge" if flags.get(f) == "edge" else "within range")
    #                    for f in FEATURES],
    #     })
    #     st.dataframe(show, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────── SENSITIVITY ──────────
with tab_sens:
    st.markdown("#### Live one-at-a-time sensitivity")
    st.caption("Every other input stays at its sidebar value while the selected "
               "feature sweeps its training range. This runs the deployed model "
               "in real time, so it always reflects the current operating point.")

    cA, cB = st.columns([1, 1])
    sweep_f = cA.selectbox("Feature to sweep", FEATURES,
                           format_func=lambda f: LABELS[f])
    n_pts = cB.slider("Sweep resolution", 15, 80, 40, 5)

    s = STATS[sweep_f]
    grid = (np.array(s["levels"]) if s["levels"]
            else np.linspace(s["min"], s["max"], n_pts))
    rows = []
    for g in grid:
        v = dict(values); v[sweep_f] = float(g)
        rows.append([v[f] for f in FEATURES])
    preds = model.predict(pd.DataFrame(rows, columns=FEATURES))

    f1 = make_subplots(rows=1, cols=2, subplot_titles=("Slope response", "Intercept response"))
    for i, (t, col) in enumerate([("Slope", PALETTE["blue"]), ("Intercept", PALETTE["red"])]):
        f1.add_trace(go.Scatter(x=grid, y=preds[:, i], mode="lines+markers",
                                line=dict(color=col, width=3), marker=dict(size=5),
                                name=t, showlegend=False), row=1, col=i + 1)
        f1.add_vline(x=values[sweep_f], line=dict(color=PALETTE["green"], dash="dot"),
                     row=1, col=i + 1)
        f1.update_xaxes(title_text=LABELS[sweep_f], row=1, col=i + 1,
                        gridcolor=PALETTE["line"])
        f1.update_yaxes(title_text=f"Predicted {t}", row=1, col=i + 1,
                        gridcolor=PALETTE["line"])
    f1.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                     font=dict(family="Inter, sans-serif", size=12.5),
                     margin=dict(l=60, r=30, t=50, b=55))
    st.plotly_chart(f1, use_container_width=True)

    st.markdown("###### Resulting family of compaction curves")
    ngc = np.arange(1, 201)
    f2 = go.Figure()
    for i, g in enumerate(grid):
        frac = i / max(len(grid) - 1, 1)
        f2.add_trace(go.Scatter(
            x=ngc, y=preds[i, 0] * np.log(ngc) + preds[i, 1], mode="lines",
            line=dict(width=1.7, color=f"rgba({int(31+190*frac)},{int(78+40*frac)},"
                                       f"{int(121-60*frac)},0.85)"),
            name=f"{LABELS[sweep_f]} = {g:.3g}", showlegend=False,
            hovertemplate=f"{LABELS[sweep_f]} = {g:.4g}<br>"
                          "N = %{x}<br>AV = %{y:.2f} %<extra></extra>"))
    f2.add_hline(y=4.0, line=dict(color=PALETTE["red"], dash="dot", width=2),
                 annotation_text="4 % target")
    f2.update_xaxes(type="log", title_text="Number of gyrations, N<sub>g</sub>",
                    gridcolor=PALETTE["line"])
    f2.update_yaxes(title_text="Air voids, AV (%)", gridcolor=PALETTE["line"])
    f2.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white",
                     font=dict(family="Inter, sans-serif", size=12.5),
                     margin=dict(l=60, r=30, t=30, b=55),
                     title=dict(text=f"Curves shaded from the lowest to the highest "
                                     f"{LABELS[sweep_f]}", font=dict(size=13)))
    st.plotly_chart(f2, use_container_width=True)

    st.markdown("###### Global sensitivity indices from the training study")
    si = pd.DataFrame(META["sensitivity_index"])
    si["Feature"] = si["Feature"].map(LABELS)
    st.dataframe(
        si.pivot(index="Feature", columns="Target", values="SI_pct")
          .round(2).rename(columns=lambda c: f"{c} - SI (%)"),
        use_container_width=True)

# ─────────────────────────────────────────────────────── BATCH ────────────────
with tab_batch:
    st.markdown("#### Predict for many mixtures at once")
    st.caption("Upload a CSV or Excel file whose columns match the model inputs. "
               "Column names must match exactly - download the template below.")

    tmpl = pd.DataFrame([{f: STATS[f]["median"] for f in FEATURES}])
    st.download_button("Download input template (CSV)",
                       tmpl.to_csv(index=False).encode(),
                       "input_template.csv", "text/csv")

    up = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"])
    if up is not None:
        try:
            raw = (pd.read_csv(up) if up.name.lower().endswith(".csv")
                   else pd.read_excel(up))
            missing = [f for f in FEATURES if f not in raw.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(map(repr, missing)))
            else:
                out = model.predict(raw[FEATURES])
                res = raw.copy()
                res["Predicted_Slope"] = out[:, 0]
                res["Predicted_Intercept"] = out[:, 1]
                res["AV_at_Ndes"] = out[:, 0] * np.log(N_des) + out[:, 1]
                res["pct_Gmm_at_Ndes"] = 100 - res["AV_at_Ndes"]
                res["Ng_at_4pct_AV"] = np.exp((4.0 - out[:, 1]) / out[:, 0])
                st.success(f"Predicted {len(res)} mixtures.")
                st.dataframe(res.head(50), use_container_width=True)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    res.to_excel(w, index=False, sheet_name="predictions")
                st.download_button("Download all predictions (XLSX)", buf.getvalue(),
                                   "batch_predictions.xlsx",
                                   "application/vnd.openxmlformats-officedocument."
                                   "spreadsheetml.sheet")

                fb = go.Figure()
                for i in range(min(len(res), 60)):
                    ngb = np.arange(1, 201)
                    fb.add_trace(go.Scatter(
                        x=ngb, y=out[i, 0] * np.log(ngb) + out[i, 1], mode="lines",
                        line=dict(width=1.2, color="rgba(31,78,121,.35)"),
                        showlegend=False, hoverinfo="skip"))
                fb.add_hline(y=4.0, line=dict(color=PALETTE["red"], dash="dot"))
                fb.update_xaxes(type="log", title_text="N<sub>g</sub>",
                                gridcolor=PALETTE["line"])
                fb.update_yaxes(title_text="AV (%)", gridcolor=PALETTE["line"])
                fb.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                                 title="Predicted curves (first 60 rows)",
                                 font=dict(family="Inter, sans-serif", size=12.5),
                                 margin=dict(l=60, r=30, t=55, b=55))
                st.plotly_chart(fb, use_container_width=True)
        except Exception as exc:                                # noqa: BLE001
            st.error(f"Could not process the file: {exc}")

# ─────────────────────────────────────────────────────── MODEL CARD ───────────
with tab_card:
    st.markdown("#### Model card")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(f"""
**Algorithm** &nbsp;·&nbsp; {META['model_name']}
**Responses** &nbsp;·&nbsp; {', '.join(META['targets'])}
**Training observations** &nbsp;·&nbsp; {META['n_observations']}
**Inputs** &nbsp;·&nbsp; {len(FEATURES)} features
**Governing equation** &nbsp;·&nbsp; `{META['equation']}`

The two responses are modelled by independent CatBoost regressors wrapped in a
`MultiOutputRegressor`; standardisation is fitted inside the pipeline, so it is
re-estimated within every cross-validation fold and no information leaks from
the validation data.
        """)
    with c2:
        st.markdown("**Input variables**")
        st.dataframe(pd.DataFrame({
            "Feature": [LABELS[f] for f in FEATURES],
            "Description": [HELP.get(f, "") for f in FEATURES],
        }), use_container_width=True, hide_index=True, height=330)

    st.markdown("##### Hold-out performance (75 / 25 split)")
    st.dataframe(pd.DataFrame(META["performance_holdout"]).round(4),
                 use_container_width=True, hide_index=True)

    st.markdown("##### Leave-one-out cross-validation")
    st.dataframe(pd.DataFrame(META["performance_loocv"]).round(4),
                 use_container_width=True, hide_index=True)

    st.markdown("##### SHAP importance (% of total)")
    shp = pd.DataFrame(META["shap_importance"])
    shp["Feature"] = shp["Feature"].map(LABELS)
    fimp = go.Figure()
    fimp.add_trace(go.Bar(y=shp["Feature"], x=shp["Slope_pct"], name="Slope",
                          orientation="h", marker_color=PALETTE["blue"]))
    fimp.add_trace(go.Bar(y=shp["Feature"], x=shp["Intercept_pct"], name="Intercept",
                          orientation="h", marker_color=PALETTE["red"]))
    fimp.update_layout(barmode="group", height=380, plot_bgcolor="white",
                       paper_bgcolor="white", xaxis_title="Mean |SHAP| (% of total)",
                       font=dict(family="Inter, sans-serif", size=12.5),
                       margin=dict(l=120, r=30, t=20, b=50),
                       legend=dict(orientation="h", y=1.06, x=0))
    fimp.update_xaxes(gridcolor=PALETTE["line"])
    st.plotly_chart(fimp, use_container_width=True)

    st.info("**Intended use.** Screening and preliminary mixture design within the "
            "range of the training data. Predictions outside that range are flagged "
            "on the Prediction tab and should not substitute for laboratory "
            "gyratory compaction testing.", icon="ℹ️")

st.markdown(
    "<div class='foot'>Superpave Gyratory Compaction Curve Predictor · "
    "CatBoost multi-output model · built with Streamlit</div>",
    unsafe_allow_html=True)
