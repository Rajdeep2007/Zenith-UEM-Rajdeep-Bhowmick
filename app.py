"""untold — SaaS Dashboard Streamlit demo app styled after modern UI design.

Run: streamlit run app.py
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from gapdetect import GapType, analyze

BASE = Path(__file__).parent
DATA = BASE / "data"

CASES = {
    p.stem: p
    for p in sorted(DATA.glob("*.json"))
    if p.is_file()
}

TYPE_META = {
    "unanswered_question": ("Unanswered Question", "#DC2626", "#FEF2F2"),
    "ignored_response": ("Ignored Response", "#EA580C", "#FFF7ED"),
    "repeated_clarification": ("Repeated Clarification", "#D97706", "#FEF3C7"),
    "unresolved_topic": ("Unresolved Topic", "#7E22CE", "#FAF5FF"),
}

SEV_COLOR = {"high": "#DC2626", "medium": "#EA580C", "low": "#16A34A"}
SEV_BG = {"high": "#FEF2F2", "medium": "#FFF7ED", "low": "#F0FDF4"}

TONE_COLOR = {
    "frustration": "#DC2626",
    "urgency": "#EA580C",
    "uncertainty": "#2563EB",
    "neutral": "#64748B",
}


def load_case(stem: str) -> tuple[str, str, str]:
    data = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    raw = "\n".join(f"{t['speaker']}: {t['text']}" for t in data["transcript"])
    return data.get("title", stem), raw, data.get("description", "")


def turn_html(turn, gap_types: list[str]) -> str:
    spk = html.escape(turn.speaker)
    txt = html.escape(turn.text)
    num = f"{turn.id}"
    if gap_types:
        label, accent, bg = TYPE_META[gap_types[0]]
        others = f" +{len(gap_types) - 1} more" if len(gap_types) > 1 else ""
        return (
            f'<div style="background:{bg};border-left:5px solid {accent};'
            f'border-radius:12px;padding:12px 16px;margin:8px 0;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.02);">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;">'
            f'<span><b style="color:#0F172A;">{spk}</b> '
            f'<span style="background:#FFFFFF;color:#64748B;font-size:0.75em;'
            f'padding:2px 8px;border-radius:10px;margin-left:6px;border:1px solid #E2E8F0;">turn #{num}</span></span>'
            f'<span style="background:{accent};color:#fff;font-size:0.75em;'
            f'font-weight:700;padding:3px 10px;border-radius:12px;">{label}{others}</span></div>'
            f'<div style="margin-top:6px;color:#334155;font-size:0.92rem;line-height:1.5;">{txt}</div></div>'
        )
    return (
        f'<div style="border-left:4px solid #CBD5E1;border-radius:12px;'
        f'padding:12px 16px;margin:8px 0;background:#FFFFFF;'
        f'border:1px solid #F1F5F9;box-shadow:0 2px 6px rgba(0,0,0,0.01);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span><b style="color:#0F172A;">{spk}</b> '
        f'<span style="background:#F1F5F9;color:#64748B;font-size:0.75em;'
        f'padding:2px 8px;border-radius:10px;margin-left:6px;">turn #{num}</span></span></div>'
        f'<div style="margin-top:6px;color:#334155;font-size:0.92rem;line-height:1.5;">{txt}</div></div>'
    )


def gap_card(g, idx: int) -> None:
    label, accent, _ = TYPE_META[g.type.value]
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 2, 2, 2])

        c1.markdown(
            f'<span style="font-size:1.05rem;font-weight:800;color:#0F172A;">{idx}. {label}</span>',
            unsafe_allow_html=True,
        )
        sev = g.severity.value.upper()
        sev_color = SEV_COLOR[g.severity.value]
        sev_bg = SEV_BG[g.severity.value]
        c2.markdown(
            f'<span style="background:{sev_bg};color:{sev_color};font-weight:700;'
            f'font-size:0.78em;padding:4px 10px;border-radius:12px;">SEVERITY: {sev}</span>',
            unsafe_allow_html=True,
        )
        tone_val = (g.tone or "neutral").lower()
        tone_color = TONE_COLOR.get(tone_val, "#64748B")
        c3.markdown(
            f'<span style="background:#F1F5F9;color:{tone_color};'
            f'font-weight:700;font-size:0.78em;padding:4px 10px;border-radius:12px;">TONE: '
            f'{tone_val.upper()}</span>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<span style="background:#F8FAFC;color:#64748B;font-size:0.8em;'
            f'font-weight:600;padding:4px 10px;border-radius:12px;border:1px solid #E2E8F0;">Turns '
            f'{g.turn_range[0]}–{g.turn_range[1]}</span>',
            unsafe_allow_html=True,
        )

        caption_bits = [
            f"<b>Participants:</b> {', '.join(g.participants)}",
            f"<b>Turn range:</b> {g.turn_range[0]}–{g.turn_range[1]}",
        ]
        if g.confidence is not None:
            pct = g.confidence * 100
            color = (
                "#10B981" if g.confidence >= 0.8
                else "#F59E0B" if g.confidence >= 0.6
                else "#64748B"
            )
            caption_bits.append(
                f'<span style="color:{color};font-weight:700;">'
                f'Confidence {pct:.0f}%</span>'
            )
        st.markdown(
            f'<div style="font-size:0.82rem;color:#64748B;margin:8px 0 6px 0;">'
            + "  ·  ".join(caption_bits) + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="font-size:0.95rem;color:#1E293B;margin-bottom:8px;font-weight:500;">{html.escape(g.description)}</div>', unsafe_allow_html=True)
        st.markdown("<b style='font-size:0.85rem;color:#0F172A;'>Verbatim Evidence:</b>", unsafe_allow_html=True)
        for q, qid in zip(g.evidence_quotes, g.message_ids):
            st.markdown(
                f'<div style="border-left:3px solid {accent};padding:6px 14px;'
                f'margin:4px 0;background:#F8FAFC;border-radius:0 8px 8px 0;'
                f'border:1px solid #F1F5F9;border-left:3px solid {accent};">'
                f'<span style="background:#E2E8F0;color:#475569;font-size:0.72em;'
                f'padding:2px 6px;border-radius:6px;font-weight:700;">#{qid}</span> '
                f'<span style="color:#334155;font-size:0.88rem;">“{html.escape(q)}”</span></div>',
                unsafe_allow_html=True,
            )
        if g.signals:
            st.markdown("<b style='font-size:0.85rem;color:#0F172A;margin-top:6px;display:block;'>Signals:</b>", unsafe_allow_html=True)
            for s in g.signals:
                st.markdown(f"<span style='color:#475569;font-size:0.85rem;'>• {html.escape(s)}</span>", unsafe_allow_html=True)

        if g.mood_delta is not None:
            word = "dip" if g.mood_delta < 0 else ("rise" if g.mood_delta > 0 else "flat")
            color = "#DC2626" if g.mood_delta < 0 else ("#10B981" if g.mood_delta > 0 else "#64748B")
            st.markdown(
                f'<div style="margin-top:8px;font-size:0.85rem;"><span style="color:{color};font-weight:700;'
                f'background:#F8FAFC;padding:4px 10px;border-radius:8px;border:1px solid #F1F5F9;">'
                f'Mood change during gap: {g.mood_delta:+.2f} ({word})</span></div>',
                unsafe_allow_html=True,
            )


def health_panel(report) -> None:
    """Gauge + sub-score bars for the Communication Health Score."""
    h = report.health
    if not h:
        return
    score = h.get("overall", 0)
    grade = h.get("grade", "?")
    color = "#10B981" if score >= 80 else ("#F59E0B" if score >= 60 else "#F95738")

    st.markdown("<h3 style='font-size:1.1rem;font-weight:800;color:#0F172A;margin-bottom:12px;'>Communication Health Breakdown</h3>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": f"Grade {grade}", "font": {"size": 18, "color": "#0F172A", "family": "Plus Jakarta Sans"}},
                number={"font": {"size": 46, "color": "#0F172A", "family": "Plus Jakarta Sans", "weight": "bold"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": color, "thickness": 0.35},
                    "steps": [
                        {"range": [0, 60], "color": "#FEF2F2"},
                        {"range": [60, 80], "color": "#FFFBEB"},
                        {"range": [80, 100], "color": "#ECFDF5"},
                    ],
                    "threshold": {
                        "line": {"color": "#0F172A", "width": 3},
                        "thickness": 0.85,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        subs = h.get("subscores", {})
        labels = {
            "responsiveness": "Responsiveness — answered questions ratio",
            "closure": "Closure — threads with resolved decisions",
            "inclusion": "Inclusion — balanced speaker participation",
            "tone": "Tone — freedom from friction/urgency cues",
        }
        for key, label in labels.items():
            val = subs.get(key, 0)
            st.progress(val / 100.0, text=f"**{val}%** · {label}")

        d = h.get("details", {})
        r = d.get("responsiveness", {})
        c = d.get("closure", {})
        i = d.get("inclusion", {})
        t = d.get("tone", {})
        st.markdown(
            f'<div style="font-size:0.78rem;color:#64748B;background:#F8FAFC;padding:10px 14px;'
            f'border-radius:10px;border:1px solid #F1F5F9;margin-top:8px;">'
            f'<b>{r.get("questions", 0)}</b> questions ({r.get("unanswered", 0)} unanswered)   ·   '
            f'<b>{c.get("closed", 0)}/{c.get("threads", 0)}</b> topics closed   ·   '
            f'<b>{i.get("speakers", 0)}</b> participants   ·   '
            f'Tone: <b>{t.get("frustration", 0)}</b> frustration / <b>{t.get("urgency", 0)}</b> urgency'
            f'</div>',
            unsafe_allow_html=True,
        )


def emotion_chart(report) -> None:
    """Per-speaker polarity over turns with gap spans shaded."""
    series = report.sentiment

    if not series:
        st.write("No sentiment data.")
        return

    fig = go.Figure()
    speakers = list(dict.fromkeys(s["speaker"] for s in series))
    palette = ["#F95738", "#8B5CF6", "#10B981", "#2563EB", "#F59E0B"]

    for idx, sp in enumerate(speakers):
        pts = [s for s in series if s["speaker"] == sp]
        line_color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=[p["turn"] for p in pts],
                y=[p["polarity"] for p in pts],
                mode="lines+markers",
                name=sp,
                line=dict(color=line_color, width=3, shape="spline"),
                marker=dict(size=7, color=line_color),
                hovertemplate="Turn %{x}<br>Polarity %{y:.2f}<extra>"
                + html.escape(sp)
                + "</extra>",
            )
        )

    for g in report.gaps:
        accent = TYPE_META[g.type.value][1]
        fig.add_vrect(
            x0=g.turn_range[0] - 0.4,
            x1=g.turn_range[1] + 0.4,
            fillcolor=accent,
            opacity=0.12,
            line_width=0,
        )
    fig.add_hline(y=0, line_color="#CBD5E1", line_dash="dot")
    fig.update_layout(
        yaxis=dict(title="Polarity", range=[-1.05, 1.05], gridcolor="#F1F5F9", title_font=dict(color="#64748B")),
        xaxis=dict(title="Turn", dtick=1, gridcolor="#F1F5F9", title_font=dict(color="#64748B")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#0F172A"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def render_kpi_cards(summary):
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-icon-circle" style="background:#FFF0EB;color:#F95738;">
                    ⚡
                </div>
                <div class="metric-val">{summary["total"]}</div>
                <div class="metric-lbl">Total Gaps Found</div>
                <div class="trend-badge trend-orange">● Live Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        high_sev = summary["by_severity"]["high"]
        trend_cls = "trend-red" if high_sev > 0 else "trend-green"
        trend_txt = f"🚨 {high_sev} High Risk" if high_sev > 0 else "✓ All Clear"
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-icon-circle" style="background:#FEF2F2;color:#DC2626;">
                    🛡️
                </div>
                <div class="metric-val">{high_sev}</div>
                <div class="metric-lbl">High Severity Gaps</div>
                <div class="trend-badge {trend_cls}">{trend_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        unans = summary["by_type"]["unanswered_question"]
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-icon-circle" style="background:#EFF6FF;color:#2563EB;">
                    ❓
                </div>
                <div class="metric-val">{unans}</div>
                <div class="metric-lbl">Unanswered Questions</div>
                <div class="trend-badge trend-neutral">In Conversation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        ign = summary["by_type"]["ignored_response"]
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-icon-circle" style="background:#FFF7ED;color:#EA580C;">
                    💬
                </div>
                <div class="metric-val">{ign}</div>
                <div class="metric-lbl">Ignored Responses</div>
                <div class="trend-badge trend-orange">Turn Overlooks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        other_cnt = summary["by_type"]["repeated_clarification"] + summary["by_type"]["unresolved_topic"]
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-icon-circle" style="background:#FAF5FF;color:#7E22CE;">
                    🔄
                </div>
                <div class="metric-val">{other_cnt}</div>
                <div class="metric-lbl">Clarif. & Unresolved</div>
                <div class="trend-badge trend-neutral">Thread Friction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="untold Dashboard",
        page_icon="🍊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #F4F5F8 !important;
            color: #0F172A;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #EAECEF !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            background-color: #F8FAFC;
            border: 1px solid #F1F5F9;
            border-radius: 10px;
            padding: 8px 14px;
            margin-bottom: 6px;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background-color: #FFF5F0;
            border-color: #FFDCD0;
            color: #F95738 !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #FFF0EB !important;
            border: 1px solid #FFCBB9 !important;
            color: #F95738 !important;
        }

        .vortex-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #FFFFFF;
            padding: 18px 24px;
            border-radius: 16px;
            border: 1px solid #EAECEF;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02);
            margin-bottom: 20px;
        }

        .vortex-header-title {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
            line-height: 1.2;
        }

        .vortex-header-subtitle {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 2px;
        }

        .vortex-header-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .search-chip {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 0.82rem;
            color: #64748B;
        }

        .user-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 4px 12px 4px 4px;
            font-size: 0.82rem;
            font-weight: 600;
            color: #1E293B;
        }

        .user-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #FF6B35, #F95738);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .metric-card-box {
            background: #FFFFFF;
            border: 1px solid #EAECEF;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.04);
        }

        .metric-icon-circle {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }

        .metric-val {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
            margin-bottom: 4px;
        }

        .metric-lbl {
            font-size: 0.78rem;
            color: #64748B;
            font-weight: 500;
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .trend-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 12px;
        }

        .trend-green { background: #ECFDF5; color: #059669; }
        .trend-red { background: #FEF2F2; color: #DC2626; }
        .trend-orange { background: #FFF7ED; color: #EA580C; }
        .trend-neutral { background: #F1F5F9; color: #64748B; }

        .stTabs [data-baseweb="tab-list"] {
            background: #FFFFFF !important;
            padding: 6px 12px !important;
            border-radius: 14px !important;
            border: 1px solid #EAECEF !important;
            gap: 8px !important;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: #64748B !important;
            border: none !important;
            padding: 0 16px !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFF0EB !important;
            color: #F95738 !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border-radius: 16px !important;
            border: 1px solid #EAECEF !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
            padding: 16px !important;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #FF6B35 0%, #F95738 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 8px 18px !important;
            box-shadow: 0 4px 12px rgba(249, 87, 56, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(249, 87, 56, 0.35) !important;
        }

        div.stDownloadButton > button {
            background: #FFFFFF !important;
            color: #F95738 !important;
            border: 1px solid #FFCBB9 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        div.stDownloadButton > button:hover {
            background: #FFF0EB !important;
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #FF6B35, #F95738) !important;
            border-radius: 8px !important;
        }

        textarea, input[type="text"] {
            border-radius: 10px !important;
            border: 1px solid #E2E8F0 !important;
            background: #FAFAFA !important;
        }

        textarea:focus, input[type="text"]:focus {
            border-color: #F95738 !important;
            box-shadow: 0 0 0 2px rgba(249, 87, 56, 0.15) !important;
        }

        .brand-logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 4px 16px 4px;
            border-bottom: 1px solid #F1F5F9;
            margin-bottom: 12px;
        }

        .brand-logo-box {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #FF6B35 0%, #F95738 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-weight: 900;
            font-size: 1.2rem;
            box-shadow: 0 4px 10px rgba(249, 87, 56, 0.3);
        }

        .brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .brand-subtitle {
            font-size: 0.72rem;
            color: #64748B;
            font-weight: 500;
        }

        .sidebar-promo-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F0 100%);
            border: 1px solid #FFDCD0;
            border-radius: 14px;
            padding: 14px;
            margin-top: 24px;
            text-align: center;
        }

        .sidebar-promo-title {
            font-weight: 700;
            font-size: 0.85rem;
            color: #0F172A;
            margin-bottom: 4px;
        }

        .sidebar-promo-desc {
            font-size: 0.75rem;
            color: #64748B;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-logo-container">
                <div class="brand-logo-box">
                    <span>🍊</span>
                </div>
                <div>
                    <div class="brand-title">untold</div>
                    <div class="brand-subtitle">Communication Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">General</div>', unsafe_allow_html=True)
        options = list(CASES) + ["Paste your own"]
        choice = st.radio("Select Conversation", options, index=0)

        st.markdown('<div class="sidebar-section-title">Analysis Settings</div>', unsafe_allow_html=True)
        window = st.slider("Reply window (turns)", 2, 6, 3)

        st.markdown('<div class="sidebar-section-title">Detector Rules</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.78rem;color:#475569;line-height:1.7;background:#F8FAFC;padding:10px 14px;border-radius:10px;border:1px solid #F1F5F9;">
                <span style="color:#DC2626;font-weight:700;">●</span> Unanswered Question<br>
                <span style="color:#EA580C;font-weight:700;">●</span> Ignored Response<br>
                <span style="color:#D97706;font-weight:700;">●</span> Repeated Clarification<br>
                <span style="color:#7E22CE;font-weight:700;">●</span> Unresolved Topic
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-promo-card">
                <div class="sidebar-promo-title">Analysis Engine</div>
                <div class="sidebar-promo-desc">Verbatim transcript verification active</div>
                <div style="background:linear-gradient(135deg, #FF6B35 0%, #F95738 100%);color:#fff;font-weight:700;font-size:0.75rem;padding:6px;border-radius:8px;box-shadow:0 4px 10px rgba(249, 87, 56, 0.25);">
                    System Ready
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Dashboard Header
    st.markdown(
        """
        <div class="vortex-header">
            <div>
                <h1 class="vortex-header-title">Dashboard</h1>
                <div class="vortex-header-subtitle">Welcome back! Here are today's reports for your team communication analysis</div>
            </div>
            <div class="vortex-header-controls">
                <div class="search-chip">
                    <span>🔍</span> Search cases... <span style="background:#E2E8F0;color:#475569;font-size:0.7em;padding:1px 6px;border-radius:4px;font-weight:700;">⌘K</span>
                </div>
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:50%;width:34px;height:34px;display:flex;align-items:center;justify-content:center;color:#64748B;font-size:0.9rem;">
                    🔔
                </div>
                <div class="user-badge">
                    <div class="user-avatar">GI</div>
                    <span>untold AI</span>
                    <span style="color:#94A3B8;font-size:0.7rem;">▼</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if choice == "Paste your own":
        sample = (
            "Alice: Can someone confirm the deployment deadline?\n"
            "Bob: I pushed the API docs to Confluence yesterday.\n"
            "Alice: Also who owns the login bug?\n"
            "Carol: Let me check Jira.\n"
        )
        raw = st.text_area("Transcript (Speaker: text)", sample, height=200)
        title, desc = "Custom transcript", ""
    else:
        title, raw, desc = load_case(choice)
        if desc:
            st.info(desc)

    report = analyze(raw, window=window, title=title)
    gap_map = report.gap_turn_ids()
    summary = report.summary

    # KPI Summary Row
    render_kpi_cards(summary)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Communication Health Section
    with st.container(border=True):
        health_panel(report)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab_transcript, tab_gaps, tab_emotion, tab_live, tab_json = st.tabs(
        ["Highlighted transcript", "Gap report", "Emotion", "Live simulation", "JSON"]
    )

    with tab_transcript:
        if not report.gaps:
            st.success(
                "No communication gaps detected — every question was answered "
                "and every topic closed."
            )
        for turn in report.transcript:
            st.markdown(
                turn_html(turn, gap_map.get(turn.id, [])), unsafe_allow_html=True
            )

    with tab_gaps:
        if not report.gaps:
            st.write("Nothing to report.")
        for i, g in enumerate(report.gaps, 1):
            gap_card(g, i)

    with tab_emotion:
        st.caption(
            "One line per participant; polarity per message (blended TextBlob "
            "sentiment + tone cues). Shaded bands are detected gaps — watch for "
            "mood dips inside them."
        )
        emotion_chart(report)
        if report.gaps:
            deltas = [
                (g, g.mood_delta) for g in report.gaps if g.mood_delta is not None
            ]
            if deltas:
                worst = min(deltas, key=lambda x: x[1])
                st.markdown(
                    f"**Largest mood drop:** {worst[0].type.value.replace('_', ' ')} "
                    f"at turns {worst[0].turn_range[0]}–{worst[0].turn_range[1]} "
                    f"(**{worst[1]:+.2f}**)"
                )

    with tab_live:
        ss = st.session_state
        n = len(report.transcript)
        if ss.get("sim_title") != title:
            ss.sim_title = title
            ss.sim_idx = 0
            ss.sim_playing = False
            ss.sim_fired = set()
        if n:
            ss.sim_idx = max(0, min(ss.sim_idx, n - 1))

        st.caption(
            "Streams the conversation turn by turn — gaps pop up as live "
            "alerts the moment they become detectable."
        )
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
        if ctrl1.button("▶ Play" if not ss.sim_playing else "⏸ Pause", key="sim_play"):
            ss.sim_playing = not ss.sim_playing
        if ctrl2.button("⏭ Step", key="sim_step"):
            ss.sim_playing = False
            ss.sim_idx = min(ss.sim_idx + 1, max(n - 1, 0))

        if ctrl3.button("⟲ Reset", key="sim_reset"):
            ss.sim_playing = False
            ss.sim_idx = 0
            ss.sim_fired = set()
        delay = ctrl4.slider("Turn delay (s)", 0.2, 2.0, 0.7, 0.1, key="sim_delay")

        if n:
            st.progress((ss.sim_idx + 1) / n, text=f"Turn {ss.sim_idx + 1} of {n}")

        revealed_idx = [
            i for i, g in enumerate(report.gaps)
            if n and g.turn_range[1] <= ss.sim_idx
        ]
        for i in revealed_idx:
            if i not in ss.sim_fired:
                g = report.gaps[i]
                label, _, _ = TYPE_META[g.type.value]
                st.toast(
                    f"{label} · turns {g.turn_range[0]}–{g.turn_range[1]}",
                    icon="🚨",
                )
                ss.sim_fired.add(i)

        stream_col, alert_col = st.columns([3, 2])
        with stream_col:
            live_map: dict[int, list[str]] = {}
            for i in revealed_idx:
                g = report.gaps[i]
                for tid in range(g.turn_range[0], g.turn_range[1] + 1):
                    live_map.setdefault(tid, []).append(g.type.value)

            for u in report.transcript[:ss.sim_idx + 1]:
                st.markdown(
                    turn_html(u, live_map.get(u.id, [])), unsafe_allow_html=True
                )
        with alert_col:
            st.markdown("**Live alerts**")
            if not revealed_idx:
                st.caption("Nothing yet — hit ▶ Play to start the stream.")
            for i in revealed_idx:
                g = report.gaps[i]
                label, accent, bg = TYPE_META[g.type.value]
                st.markdown(
                    f'<div style="border-left:4px solid {accent};background:{bg};'
                    f'padding:8px 12px;border-radius:0 8px 8px 0;margin:6px 0;">'
                    f'<b>{label}</b> · turns {g.turn_range[0]}–{g.turn_range[1]} · '
                    f'<span style="color:{SEV_COLOR[g.severity.value]};">'
                    f'{g.severity.value}</span></div>',
                    unsafe_allow_html=True,
                )

        if ss.sim_playing:
            if n == 0:
                ss.sim_playing = False
            elif ss.sim_idx < n - 1:
                time.sleep(delay)
                ss.sim_idx += 1
                st.rerun()
            else:
                ss.sim_playing = False
                st.success("End of conversation — all gaps surfaced.")

    with tab_json:
        st.json(report.to_dict())
        st.download_button(
            "Download report JSON",
            data=report.to_json(),
            file_name=f"{title.lower().replace(' ', '_')}_gaps.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
