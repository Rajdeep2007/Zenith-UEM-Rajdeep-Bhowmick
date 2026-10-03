"""GapSense — Streamlit demo app.

Run:  streamlit run app.py
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
DATA =BASE /"data"

CASES ={
    p.stem :p
    for p in sorted (DATA.glob("*.json" ))
    if p.is_file()
}

TYPE_META= {
    "unanswered_question" : ("Unanswered Question","#c62828","#ffebee") ,
    "ignored_response" : ("Ignored Response", "#ef6c00", "#fff3e0" ) ,
    "repeated_clarification": ("Repeated Clarification", "#f9a825", "#fffde7"),
    "unresolved_topic" : ( "Unresolved Topic","#6a1b9a", "#f3e5f5" ) ,
}

SEV_COLOR={ "high": "#c62828" , "medium":"#ef6c00","low":"#2e7d32"}
TONE_COLOR = {
    "frustration" :"#c62828" ,
    "urgency" :"#ef6c00",
    "uncertainty": "#1565c0",
    "neutral": "#616161",
}


def load_case(stem: str) -> tuple[str, str, str]:
    data = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    raw = "\n".join (f"{t['speaker']}: {t['text']}" for t in data["transcript"] )


    return data.get("title", stem), raw, data.get("description", "")


def turn_html(turn, gap_types: list[str]) -> str:
    spk= html.escape ( turn.speaker )
    txt=html.escape(turn.text )
    num = f"{turn.id}"
    if gap_types :

        label, accent , bg = TYPE_META [gap_types[0] ]
        others = (
            f" +{len(gap_types) - 1} more" if len(gap_types) > 1 else ""
        )
        return (
            f'<div style="background:{bg};border-left:5px solid {accent};'
            f'border-radius:8px;padding:10px 14px;margin:6px 0;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;">'
            f'<span><b>{spk}</b> '
            f'<span style="color:#9e9e9e;font-size:0.82em;">turn {num}</span></span>'
            f'<span style="background:{accent};color:#fff;font-size:0.75em;'
            f'padding:2px 10px;border-radius:12px;">{label}{others}</span></div>'
            f'<div style="margin-top:4px;">{txt}</div></div>'
        )
    return(
        f'<div style="border-left:5px solid #e0e0e0;border-radius:8px;'
        f'padding:10px 14px;margin:6px 0;background:#fafafa;">'
        f'<span><b>{spk}</b> '
        f'<span style="color:#9e9e9e;font-size:0.82em;">turn {num}</span></span>'
        f'<div style="margin-top:4px;">{txt}</div></div>'
    )


def gap_card( g ,idx :int) -> None:
    label, accent, _ = TYPE_META[g.type.value]
    with st.container ( border =True) :
        c1,c2,c3 ,c4 =st.columns ( [4, 2 ,2 ,2 ] )

        c1.markdown(f"**{idx}. {label}**")
        sev=g.severity.value.upper( )
        c2.markdown (
            f'<span style="color:{SEV_COLOR[g.severity.value]};font-weight:700;'
            f'font-size:0.85em;">SEVERITY: {sev}</span>',
            unsafe_allow_html=True,
        )
        c3.markdown (
            f'<span style="color:{TONE_COLOR.get(g.tone, "#616161")};'
            f'font-weight:700;font-size:0.85em;">TONE: '
            f'{(g.tone or "neutral").upper()}</span>',
            unsafe_allow_html= True ,
        )
        c4.markdown(
            f'<span style="color:#616161;font-size:0.85em;">turns '
            f'{g.turn_range[0]}–{g.turn_range[1]}</span>',
            unsafe_allow_html=True,
        )
        caption_bits = [
            f"Participants: {', '.join(g.participants)}",
            f"Turn range: {g.turn_range[0]}–{g.turn_range[1]}",
        ]
        if g.confidence is not None :
            pct = g.confidence * 100
            color =(
                "#2e7d32" if g.confidence>=0.8
                else "#f9a825" if g.confidence>= 0.6
                else "#9e9e9e"
            )
            caption_bits.append (
                f'<span style="color:{color};font-weight:700;">'
                f'model confidence {pct:.0f}%</span>'
            )
        st.caption("  ·  ".join ( caption_bits), unsafe_allow_html=True )
        st.write(g.description )
        st.markdown ( "**Evidence:**")
        st.caption (
            "Cited messages: "
            + " ".join (f"`#{i}`" for i in g.message_ids)
            + "  ·  quotes verified verbatim against cited messages"
        )
        for q,qid in zip (g.evidence_quotes , g.message_ids):
            st.markdown(
                f'<div style="border-left:3px solid {accent};padding:4px 12px;'
                f'margin:4px 0;background:#f7f7f7;border-radius:0 6px 6px 0;">'
                f'<span style="color:#9e9e9e;font-size:0.75em;">#{qid}</span> '
                f'“{html.escape(q)}”</div>',
                unsafe_allow_html=True,
            )
        if g.signals:
            st.markdown( "**Signals:**")
            for s in g.signals:
                st.markdown(f"- {s}")

        if g.mood_delta is not None:
            word = "dip" if g.mood_delta < 0 else ("rise" if g.mood_delta > 0 else "flat")
            color = "#c62828" if g.mood_delta < 0 else ("#2e7d32" if g.mood_delta > 0 else "#616161")
            st.markdown(
                f'<span style="color:{color};font-weight:600;">'
                f'Mood change during this gap: {g.mood_delta:+.2f} ({word})</span>',
                unsafe_allow_html=True,
            )


def health_panel( report )->None:
    """Gauge + sub-score bars for the Communication Health Score."""
    h= report.health
    if not h:
        return
    score = h.get("overall", 0)
    grade = h.get("grade", "?")
    color= "#2e7d32" if score >= 80 else("#ef6c00" if score>= 60 else "#c62828" )

    st.subheader ( "Communication Health Score" )
    left ,right=st.columns ([ 1,2] )



    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value= score ,
                title ={"text": f"Grade {grade}", "font":{"size" : 18 }} ,
                number={"font": {"size": 44}},
                gauge = {
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":{ "color" :color , "thickness": 0.35 } ,
                    "steps": [
                        { "range" :[0, 60], "color" :"#ffebee" } ,
                        {"range": [ 60, 80 ], "color": "#fff8e1"} ,
                        { "range":[ 80, 100] , "color": "#e8f5e9"} ,
                    ],
                    "threshold" :{
                        "line": {"color": "#424242" , "width" :3 } ,
                        "thickness" :0.85,
                        "value":score ,
                    } ,
                } ,
            )
        )
        fig.update_layout(height=280 , margin =dict(l = 30 , r = 30,t=60 , b =10 ) )
        st.plotly_chart ( fig, width = "stretch" )

    with right:
        subs= h.get ("subscores" ,{})
        labels = {
            "responsiveness": "Responsiveness — questions that got answers",
            "closure" :"Closure — topics ended with a decision",
            "inclusion": "Inclusion — balanced participation",
            "tone" :"Tone — freedom from frustration/urgency cues",
        }
        for key, label in labels.items():
            val = subs.get(key, 0)
            st.progress( val ,text=f"**{val}** · {label}")

        d =h.get( "details" , { })
        r = d.get ("responsiveness" ,{ })
        c=d.get ( "closure" , { } )
        i=d.get ("inclusion", {})
        t=d.get ("tone",{ })
        st.caption (
            f"{r.get('questions', 0)} questions asked, "
            f"{r.get('unanswered', 0)} unanswered   ·   "
            f"{c.get('closed', 0)}/{c.get('threads', 0)} topic threads closed   ·   "
            f"{i.get('speakers', 0)} participants, {i.get('ignored', 0)} ignored turns   ·   "
            f"tone cues: {t.get('frustration', 0)} frustration / "
            f"{t.get('urgency', 0)} urgency / {t.get('uncertainty', 0)} uncertainty"
        )


def emotion_chart ( report )-> None :
    """Per-speaker polarity over turns with gap spans shaded."""
    series = report.sentiment

    if not series:
        st.write("No sentiment data.")
        return

    fig = go.Figure ()
    speakers = list(dict.fromkeys(s ["speaker" ]for s in series))
    for sp in speakers:
        pts =[ s for s in series if s[ "speaker"] ==sp]
        fig.add_trace (
            go.Scatter(
                x = [ p ["turn"]for p in pts ],
                y= [p[ "polarity" ] for p in pts ] ,
                mode = "lines+markers",
                name= sp,
                hovertemplate= "turn %{x}<br>polarity %{y:.2f}<extra>"
                + html.escape (sp )
                + "</extra>" ,
            )
        )

    for g in report.gaps :
        accent=TYPE_META[g.type.value][ 1]
        fig.add_vrect (
            x0= g.turn_range [0]- 0.4,
            x1=g.turn_range[1] + 0.4,
            fillcolor= accent ,
            opacity = 0.14,
            line_width =0 ,
        )
    fig.add_hline(y=0, line_color="#bdbdbd", line_dash="dot")
    fig.update_layout (
        yaxis = dict (title="polarity",range =[- 1.05, 1.05]) ,
        xaxis =dict (title ="turn",dtick = 1) ,
        legend_title = "speaker",
        height=400,
        margin=dict ( l=20 ,r=20,t =30,b= 20 ),
        hovermode ="x unified",
    )
    st.plotly_chart ( fig , width="stretch")


def main() -> None:
    st.set_page_config (
        page_title = "GapSense",
        layout="wide",
    )
    st.title ("GapSense" )
    st.caption(
        "Detects unanswered questions, ignored responses, repeated "
        "clarifications and unresolved topics — with verbatim evidence."
    )

    with st.sidebar :

        st.header ( "Input")
        options=list (CASES ) +[ "Paste your own" ]
        choice= st.radio( "Conversation",options ,index =0)
        window=st.slider("Reply window (turns)" ,2, 6 ,3)
        st.divider ( )
        st.markdown(
            "**Gap types**\n"
            "- Unanswered Question\n"
            "- Ignored Response\n"
            "- Repeated Clarification\n"
            "- Unresolved Topic"
        )

    if choice == "Paste your own":
        sample = (
            "Alice: Can someone confirm the deployment deadline?\n"
            "Bob: I pushed the API docs to Confluence yesterday.\n"
            "Alice: Also who owns the login bug?\n"
            "Carol: Let me check Jira.\n"
        )
        raw= st.text_area ( "Transcript (Speaker: text)", sample , height=260 )
        title ,desc="Custom transcript",""
    else:


        title ,raw,desc=load_case(choice )

        if desc:
            st.info (desc)

    report = analyze (raw, window= window ,title =title )
    gap_map= report.gap_turn_ids ( )
    summary = report.summary



    #---------- summary ----------
    st.divider ( )
    m1, m2, m3 ,m4 , m5 =st.columns( 5)
    m1.metric ( "Gaps found", summary [ "total"])
    m2.metric("High severity", summary["by_severity"]["high"])
    m3.metric("Unanswered", summary["by_type"]["unanswered_question"])
    m4.metric ("Ignored" ,summary[ "by_type"]["ignored_response" ] )
    m5.metric (
        "Clarif. / Unresolved" ,
        summary["by_type"]["repeated_clarification"]
        + summary["by_type"]["unresolved_topic"],
    )

    health_panel(report)

    st.divider ( )

    tab_transcript, tab_gaps, tab_emotion, tab_live, tab_json = st.tabs(
        [ "Highlighted transcript" ,"Gap report","Emotion", "Live simulation", "JSON"]
    )

    with tab_transcript :
        if not report.gaps:
            st.success(
                "No communication gaps detected — every question was answered "
                "and every topic closed."
            )
        for turn in report.transcript :
            st.markdown(
                turn_html( turn ,gap_map.get(turn.id,[ ] ) ) , unsafe_allow_html= True
            )

    with tab_gaps :
        if not report.gaps:


            st.write ( "Nothing to report." )
        for i, g in enumerate(report.gaps, 1):
            gap_card( g,i)

    with tab_emotion :
        st.caption (
            "One line per participant; polarity per message (blended TextBlob "
            "sentiment + tone cues). Shaded bands are detected gaps — watch for "
            "mood dips inside them."
        )
        emotion_chart (report )
        if report.gaps:
            deltas=[
                (g, g.mood_delta) for g in report.gaps if g.mood_delta is not None
            ]
            if deltas :
                worst = min(deltas, key=lambda x: x[1])
                st.markdown(
                    f"**Largest mood drop:** {worst[0].type.value.replace('_', ' ')} "
                    f"at turns {worst[0].turn_range[0]}–{worst[0].turn_range[1]} "
                    f"(**{worst[1]:+.2f}**)"
                )

    with tab_live:
        ss = st.session_state
        n= len (report.transcript)
        if ss.get( "sim_title") !=title :
            ss.sim_title= title
            ss.sim_idx=0

            ss.sim_playing = False
            ss.sim_fired= set( )
        if n :
            ss.sim_idx = max ( 0 ,min (ss.sim_idx ,n -1 ) )

        st.caption (
            "Streams the conversation turn by turn — gaps pop up as live "
            "alerts the moment they become detectable."
        )
        ctrl1 ,ctrl2 , ctrl3, ctrl4 =st.columns ( [1,1 , 1,2])
        if ctrl1.button("▶ Play" if not ss.sim_playing else "⏸ Pause", key="sim_play"):
            ss.sim_playing = not ss.sim_playing
        if ctrl2.button("⏭ Step", key="sim_step"):
            ss.sim_playing =False

            ss.sim_idx = min(ss.sim_idx + 1, max(n - 1, 0))

        if ctrl3.button("⟲ Reset", key="sim_reset"):
            ss.sim_playing =False
            ss.sim_idx = 0
            ss.sim_fired = set()
        delay = ctrl4.slider( "Turn delay (s)" ,0.2, 2.0 ,0.7,0.1, key="sim_delay" )

        if n:
            st.progress ((ss.sim_idx+ 1) / n , text =f"Turn {ss.sim_idx + 1} of {n}")

        revealed_idx= [
            i for i,g in enumerate(report.gaps )
            if n and g.turn_range[1 ]<= ss.sim_idx
        ]
        for i in revealed_idx:
            if i not in ss.sim_fired :
                g = report.gaps[i]
                label, _, _ = TYPE_META[g.type.value]
                st.toast(
                    f"{label} · turns {g.turn_range[0]}–{g.turn_range[1]}",
                    icon ="🚨",
                )
                ss.sim_fired.add (i)

        stream_col, alert_col = st.columns([3, 2])
        with stream_col:
            live_map : dict [ int , list[str ]]={}
            for i in revealed_idx :
                g=report.gaps [i ]
                for tid in range (g.turn_range[ 0] ,g.turn_range [ 1 ] + 1 ):
                    live_map.setdefault( tid ,[ ]).append (g.type.value)

            for u in report.transcript[ :ss.sim_idx+ 1] :
                st.markdown (
                    turn_html ( u,live_map.get ( u.id,[] ) ), unsafe_allow_html=True
                )
        with alert_col :
            st.markdown("**Live alerts**" )
            if not revealed_idx :
                st.caption("Nothing yet — hit ▶ Play to start the stream.")
            for i in revealed_idx:
                g = report.gaps[i]
                label,accent ,bg = TYPE_META[g.type.value]
                st.markdown(
                    f'<div style="border-left:4px solid {accent};background:{bg};'
                    f'padding:8px 12px;border-radius:0 8px 8px 0;margin:6px 0;">'
                    f'<b>{label}</b> · turns {g.turn_range[0]}–{g.turn_range[1]} · '
                    f'<span style="color:{SEV_COLOR[g.severity.value]};">'
                    f'{g.severity.value}</span></div>',
                    unsafe_allow_html = True,
                )

        if ss.sim_playing:
            if n == 0:
                ss.sim_playing = False
            elif ss.sim_idx < n - 1:
                time.sleep (delay)
                ss.sim_idx += 1
                st.rerun()
            else :


                ss.sim_playing= False
                st.success("End of conversation — all gaps surfaced.")

    with tab_json:
        st.json (report.to_dict())
        st.download_button (
            "Download report JSON",
            data=report.to_json(),
            file_name =f"{title.lower().replace(' ', '_')}_gaps.json",
            mime= "application/json" ,
        )


if __name__ =="__main__":
    main ()
