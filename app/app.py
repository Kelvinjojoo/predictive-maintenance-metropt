import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_DIR = os.path.join(APP_DIR, "data")
SCORES_FILE = os.path.join(RESULTS_DIR, "window_scores.parquet")
EVENTS_FILE = os.path.join(RESULTS_DIR, "alarm_events.csv")
LEAKS_FILE = os.path.join(RESULTS_DIR, "leaks.csv")
RULES_FILE = os.path.join(RESULTS_DIR, "alarm_rules.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "test_results.csv")
TRADEOFF_FILE = os.path.join(RESULTS_DIR, "tradeoff.csv")
READINGS_FILE = os.path.join(DATA_DIR, "readings.parquet")
PARTS_FILE = os.path.join(DATA_DIR, "forest_parts.parquet")
AIR_SYSTEM_IMAGE = os.path.join(ROOT_DIR, "images", "air_system.png")

MODEL_NAMES = {"isolation_forest": "Isolation Forest", "lstm_autoencoder": "LSTM autoencoder"}
MODEL_COLORS = {"isolation_forest": "#1D9E75", "lstm_autoencoder": "#534AB7"}
VALUE_NAMES = {"isolation_forest": "score", "lstm_autoencoder": "error"}
ZONE_COLORS = {"warning": "#EF9F27", "leak": "#E24B4A", "ignored": "#B4B2A9"}
ALARM_COLOR = "#E24B4A"
INK = "#1F1F1E"
MUTED = "#5F5E5A"
LANES = [("Reservoirs", "Reservoirs: tank pressure (bar)", "#2C2C2A", 160), ("DV_eletric", "DV_eletric: the pump is pumping", "#8A8983", 62),
         ("DV_pressure", "DV_pressure: air dryer (bar)", "#185FA5", 112)]
SIGNAL_LABELS = {"TP2": "compressor", "TP3": "panel", "H1": "filter", "DV_pressure": "air dryer", "Reservoirs": "tanks", "Oil_temperature": "oil",
                 "Motor_current": "motor", "COMP": "intake valve", "DV_eletric": "outlet valve", "Towers": "dryer towers", "MPG": "start switch",
                 "Pressure_switch": "dryer switch", "pump cycles": ""}
STEP_MINUTES = 5
SCENE_MARGIN_HOURS = 12
DEFAULT_MOMENT = 4
PLAY_SECONDS = 1.0
SPEEDS = {"15 min per second": 3, "1 hour per second": 12, "3 hours per second": 36}
TOP_SIGNALS = 4

st.set_page_config(page_title="Air Leak Watch", page_icon=":material/air:", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1280px;}
.lw-brand {font-size: 0.85rem; color: #5F5E5A; margin-bottom: -0.4rem;}
.lw-brand b {color: #1F1F1E;}
.lw-lead {color: #5F5E5A; font-size: 1rem; margin-top: -0.6rem;}
.lw-label {font-size: 0.78rem; color: #5F5E5A; margin-bottom: 0.15rem;}
.lw-clock {font-size: 1.9rem; font-weight: 650; letter-spacing: -0.01em; line-height: 1.15; color: #1F1F1E;}
.lw-note {font-size: 0.9rem; color: #3D3D3A; margin-top: 0.45rem; display: flex; gap: 0.45rem; align-items: center;}
.lw-dot {width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none;}
.lw-model {display: flex; align-items: center; gap: 0.5rem; font-weight: 600; color: #1F1F1E;}
.lw-state {display: flex; align-items: center; gap: 0.6rem; margin: 0.55rem 0 0.35rem 0;}
.lw-lamp {width: 18px; height: 18px; border-radius: 50%; flex: none;}
.lw-lamp.on {background: #E24B4A; box-shadow: 0 0 0 5px rgba(226, 75, 74, 0.16), 0 0 16px rgba(226, 75, 74, 0.6);}
.lw-lamp.off {border: 2px solid #B4B2A9;}
.lw-word {font-size: 1.35rem; font-weight: 700; letter-spacing: 0.01em;}
.lw-word.on {color: #C73B3A;}
.lw-word.off {color: #8A8983;}
.lw-since {font-size: 0.85rem; color: #5F5E5A;}
.lw-value {font-size: 0.9rem; color: #3D3D3A; margin-bottom: 0.35rem;}
.lw-gauge {position: relative; height: 8px; border-radius: 4px; background: #ECECE8;}
.lw-fill {height: 8px; border-radius: 4px;}
.lw-limit {position: absolute; left: 50%; top: -4px; width: 2px; height: 16px; background: #1F1F1E;}
.lw-gauge-text {display: flex; justify-content: space-between; font-size: 0.72rem; color: #8A8983; margin-top: 0.2rem;}
.lw-why {display: grid; grid-template-columns: 11rem 1fr 2.6rem; gap: 0.3rem 0.6rem; align-items: center; font-size: 0.86rem;}
.lw-why .bar {height: 9px; border-radius: 3px;}
.lw-why .pct {text-align: right; color: #3D3D3A; font-variant-numeric: tabular-nums;}
.lw-why .name {color: #1F1F1E;}
.lw-why .name span {color: #8A8983;}
.lw-big {font-size: 2.2rem; font-weight: 700; line-height: 1.1; color: #1F1F1E;}
.lw-step {font-size: 0.8rem; font-weight: 700; color: #534AB7;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    scores = pd.read_parquet(SCORES_FILE)
    with open(RULES_FILE, encoding="utf-8") as f:
        rules = json.load(f)
    return {"test": scores[scores["split"] == "test"].reset_index(drop=True),
            "parts": pd.read_parquet(PARTS_FILE),
            "readings": pd.read_parquet(READINGS_FILE),
            "events": pd.read_csv(EVENTS_FILE, parse_dates=["first_alarm", "last_alarm"]),
            "leaks": pd.read_csv(LEAKS_FILE, parse_dates=["warning_from", "start", "end", "ignored_until"], keep_default_na=False),
            "rules": rules,
            "summary": pd.read_csv(SUMMARY_FILE, index_col="model"),
            "tradeoff": pd.read_csv(TRADEOFF_FILE)}


def check_files():
    missing = [os.path.relpath(path, ROOT_DIR) for path in [SCORES_FILE, EVENTS_FILE, LEAKS_FILE, RULES_FILE, SUMMARY_FILE, TRADEOFF_FILE, READINGS_FILE, PARTS_FILE]
               if not os.path.exists(path)]
    if missing:
        st.error("Some files of the app are missing: " + ", ".join(missing) + ". Run the notebooks up to 6.1 first (see the README).")
        st.stop()


def make_scenes(leaks, events, margin_hours):
    margin = pd.Timedelta(hours=margin_hours)
    scenes = {}
    for row in leaks.itertuples():
        scenes[f"Leak {row.leak}, reported {row.start:%d %b}"] = (row.warning_from - margin, row.end + margin)
    false_alarms = events[events["kind"] == "false alarm"]
    for kind, group in false_alarms.groupby("looks_like", sort=False):
        biggest = group.loc[group["alarm_windows"].idxmax()]
        scenes[f"False alarm: {kind}, {biggest['first_alarm']:%d %b}"] = (biggest["first_alarm"] - margin, biggest["last_alarm"] + margin)
    return scenes


def in_utc(frame, column):
    return frame.assign(**{column: frame[column].dt.tz_localize("UTC")})


def vega_time(stamp):
    return alt.DateTime(year=stamp.year, month=stamp.month, date=stamp.day, hours=stamp.hour, minutes=stamp.minute, utc=True)


def time_ticks(scene_from, scene_to):
    hours = (scene_to - scene_from) / pd.Timedelta(hours=1)
    step = 3 if hours <= 36 else 6 if hours <= 80 else 12
    margin = pd.Timedelta(hours=hours * 0.06)
    ticks = pd.date_range(scene_from.ceil(f"{step}h"), scene_to, freq=f"{step}h")
    return [vega_time(stamp=tick) for tick in ticks if scene_from + margin <= tick <= scene_to - margin]


def time_axis(column, scene_from, scene_to, show_labels):
    return alt.X(f"{column}:T", scale=alt.Scale(type="utc", domain=[vega_time(stamp=scene_from), vega_time(stamp=scene_to)]),
                 axis=alt.Axis(title=None, format="%d %b %H:%M", labelAngle=0, values=time_ticks(scene_from=scene_from, scene_to=scene_to),
                               labels=show_labels, grid=False, domainColor="#C9CBC6", tickColor="#C9CBC6", labelColor=MUTED))


def y_axis(title, show_labels, values):
    return alt.Axis(title=title, titleAngle=0, titleAlign="left", titleBaseline="bottom", titleX=-46, titleY=-6, titleFontWeight="normal",
                    titleColor="#3D3D3A", titleFontSize=11.5, labelColor=MUTED, labels=show_labels, minExtent=46, maxExtent=46,
                    gridColor="#EFEFEC", grid=show_labels, domain=False, ticks=False, tickCount=4, values=alt.Undefined if values is None else values)


def zone_rows(leaks, scene_from, scene_to):
    rows = []
    for row in leaks.itertuples():
        for zone, start, end in [("warning", row.warning_from, row.start), ("leak", row.start, row.end), ("ignored", row.end, row.ignored_until)]:
            if end > scene_from and start < scene_to:
                rows.append({"start": max(start, scene_from), "end": min(end, scene_to), "zone": zone})
    return pd.DataFrame(rows, columns=["start", "end", "zone"])


def background_layers(zones, scene_from, scene_to, show_zones, show_labels):
    if not show_zones or zones.empty:
        return []
    return [alt.Chart(in_utc(frame=in_utc(frame=zones, column="start"), column="end")).mark_rect(opacity=0.16).encode(
        x=time_axis(column="start", scene_from=scene_from, scene_to=scene_to, show_labels=show_labels), x2="end:T",
        color=alt.Color("zone:N", scale=alt.Scale(domain=list(ZONE_COLORS), range=list(ZONE_COLORS.values())), legend=None))]


def cursor_layer(cursor, scene_from, scene_to, show_labels):
    frame = in_utc(frame=pd.DataFrame({"t": [cursor]}), column="t")
    return alt.Chart(frame).mark_rule(color=INK, strokeWidth=1.4).encode(x=time_axis(column="t", scene_from=scene_from, scene_to=scene_to, show_labels=show_labels))


def compressor_lanes(readings, zones, scene_from, scene_to, cursor, show_zones):
    scene = readings[(readings["minute"] >= scene_from) & (readings["minute"] <= scene_to)]
    shown = scene[scene["minute"] <= cursor].copy()
    shown["run"] = (shown["minute"].diff() > pd.Timedelta(minutes=1)).cumsum()
    shown = in_utc(frame=shown, column="minute")
    domains = {"Reservoirs": [float(np.floor(scene["Reservoirs"].min() - 0.3)), float(np.ceil(scene["Reservoirs"].max() + 0.2))],
               "DV_eletric": [0, 1], "DV_pressure": [-0.2, float(max(2.5, np.ceil(scene["DV_pressure"].max() + 0.2)))]}
    lanes = []
    for position, (signal, title, color, height) in enumerate(LANES):
        labels = position == len(LANES) - 1
        x = time_axis(column="minute", scene_from=scene_from, scene_to=scene_to, show_labels=labels)
        if signal == "DV_eletric":
            mark = alt.Chart(shown).mark_area(color=color, opacity=0.85, interpolate="step-after", line=False)
        else:
            mark = alt.Chart(shown).mark_line(color=color, strokeWidth=1.1)
        line = mark.encode(x=x, y=alt.Y(f"{signal}:Q", scale=alt.Scale(domain=domains[signal], clamp=True),
                                        axis=y_axis(title=title, show_labels=signal != "DV_eletric", values=None)),
                           detail="run:N", tooltip=[alt.Tooltip("minute:T", title="time", format="%d %b %H:%M", formatType="utc"),
                                                    alt.Tooltip(f"{signal}:Q", title=title, format=".2f")])
        layers = background_layers(zones=zones, scene_from=scene_from, scene_to=scene_to, show_zones=show_zones, show_labels=labels)
        lanes.append(alt.layer(*layers, line, cursor_layer(cursor=cursor, scene_from=scene_from, scene_to=scene_to, show_labels=labels))
                     .properties(height=height).configure_view(stroke=None))
    return lanes


def model_lanes(test, zones, limits, scene_from, scene_to, cursor, show_zones):
    scene = test[(test["end"] >= scene_from) & (test["end"] <= scene_to)]
    shown = scene[scene["end"] <= cursor].copy()
    shown["run"] = (shown["end"].diff() > pd.Timedelta(minutes=STEP_MINUTES)).cumsum()
    shown = in_utc(frame=shown, column="end")
    lanes = []
    for model, height, labels in [("isolation_forest", 132, False), ("lstm_autoencoder", 152, True)]:
        values = pd.concat([scene[model], pd.Series([limits[model]])])
        if model == "lstm_autoencoder":
            scale = alt.Scale(type="log", domain=[max(0.05, float(values.min()) * 0.8), float(values.max()) * 1.25], nice=False)
            ticks = [0.1, 0.2, 0.5, 1, 2, 4]
        else:
            scale = alt.Scale(domain=[float(values.min()) - 0.01, float(values.max()) + 0.01], zero=False)
            ticks = None
        x = time_axis(column="end", scene_from=scene_from, scene_to=scene_to, show_labels=labels)
        title = f"{MODEL_NAMES[model]}: {VALUE_NAMES[model]}"
        line = alt.Chart(shown).mark_line(color=MODEL_COLORS[model], strokeWidth=1.4).encode(
            x=x, y=alt.Y(f"{model}:Q", scale=scale, axis=y_axis(title=title, show_labels=True, values=ticks)), detail="run:N",
            tooltip=[alt.Tooltip("end:T", title="window end", format="%d %b %H:%M", formatType="utc"), alt.Tooltip(f"{model}:Q", title=title, format=".3f")])
        alarms = alt.Chart(shown[shown[f"{model}_alarm"]]).mark_circle(color=ALARM_COLOR, size=22, opacity=1).encode(x=x, y=alt.Y(f"{model}:Q", scale=scale))
        limit = alt.Chart(pd.DataFrame({"limit": [limits[model]]})).mark_rule(color=INK, strokeDash=[5, 4], strokeWidth=1).encode(y=alt.Y("limit:Q", scale=scale))
        layers = background_layers(zones=zones, scene_from=scene_from, scene_to=scene_to, show_zones=show_zones, show_labels=labels)
        lanes.append(alt.layer(*layers, limit, line, alarms, cursor_layer(cursor=cursor, scene_from=scene_from, scene_to=scene_to, show_labels=labels))
                     .properties(height=height).configure_view(stroke=None))
    return lanes


def hours_and_minutes(duration):
    minutes = int(duration / pd.Timedelta(minutes=1))
    return f"{minutes // 60} h {minutes % 60:02d} min"


def zone_note(cursor, leaks):
    for row in leaks.itertuples():
        if row.warning_from <= cursor < row.start:
            return ZONE_COLORS["warning"], f"Warning zone: leak {row.leak} starts in {hours_and_minutes(duration=row.start - cursor)}"
        if row.start <= cursor <= row.end:
            return ZONE_COLORS["leak"], f"Leak {row.leak} is running (reported)"
        if row.end < cursor <= row.ignored_until:
            return ZONE_COLORS["ignored"], f"After leak {row.leak}: ignored zone"
    return "#C9CBC6", "Healthy time: no leak reported"


def lamp_card(model, window, limit, events, cursor):
    name, color = MODEL_NAMES[model], MODEL_COLORS[model]
    if window is None:
        state, word, since, value_line, fill = "off", "no data", "", "No window at this time", 0
    else:
        alarm = bool(window[f"{model}_alarm"])
        state, word = ("on", "ALARM") if alarm else ("off", "quiet")
        own = events[(events["model"] == name) & (events["first_alarm"] <= cursor) & (events["last_alarm"] >= cursor)]
        since = f"alarm event since {own['first_alarm'].iloc[0]:%d %b %H:%M}" if alarm and len(own) else ""
        value = float(window[model])
        value_line = f"{VALUE_NAMES[model]} <b>{value:.3f}</b> &nbsp;limit {limit:.3f}"
        fill = min(value / (2 * limit), 1) * 100
    return f"""
<div class="lw-model"><span class="lw-dot" style="background:{color}"></span>{name}</div>
<div class="lw-state"><span class="lw-lamp {state}"></span><span class="lw-word {state}">{word}</span><span class="lw-since">{since}</span></div>
<div class="lw-value">{value_line}</div>
<div class="lw-gauge"><div class="lw-fill" style="width:{fill:.0f}%; background:{color}"></div><div class="lw-limit"></div></div>
<div class="lw-gauge-text"><span>0</span><span>limit</span><span>2 x limit</span></div>
"""


def why_rows(parts, color):
    if parts.sum() <= 0:
        return '<div class="lw-since">Nothing pushes this window towards unusual.</div>'
    top = (parts / parts.sum()).sort_values(ascending=False).head(TOP_SIGNALS)
    rows = []
    for signal, share in top.items():
        rows.append(f'<div class="name">{signal} <span>{SIGNAL_LABELS.get(signal, "")}</span></div>'
                    f'<div><div class="bar" style="width:{max(share, 0) / top.max() * 100:.0f}%; background:{color}"></div></div>'
                    f'<div class="pct">{share:.0%}</div>')
    return '<div class="lw-why">' + "".join(rows) + "</div>"


def page_header(title, lead):
    st.markdown('<div class="lw-brand"><b>Air Leak Watch</b> &nbsp;Isolation Forest against an LSTM autoencoder on a metro train compressor</div>',
                unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lw-lead">{lead}</div>', unsafe_allow_html=True)


def toggle_play(scene_from, scene_to):
    if st.session_state.playing:
        st.session_state.playing = False
    else:
        if st.session_state.cursor >= scene_to.to_pydatetime():
            st.session_state.cursor = scene_from.to_pydatetime()
        st.session_state.playing = True


def replay_page():
    data = load_data()
    test, parts, readings, events, leaks = data["test"], data["parts"], data["readings"], data["events"], data["leaks"]
    limits = {model: data["rules"]["limits"][model]["main"] for model in MODEL_NAMES}
    scenes = make_scenes(leaks=leaks, events=events, margin_hours=SCENE_MARGIN_HOURS)
    names = list(scenes)
    page_header(title="Replay", lead="Watch the test months again, five minutes at a time, the way both models saw them. Pick a moment and press Play.")

    if "scene" not in st.session_state:
        wanted = st.query_params.get("scene", str(DEFAULT_MOMENT))
        st.session_state.scene = names[min(max(int(wanted), 1), len(names)) - 1] if wanted.isdigit() else names[DEFAULT_MOMENT - 1]
    controls = st.columns([2.8, 1.1, 1.7, 1.9], vertical_alignment="bottom")
    scene_name = controls[0].selectbox("Moment", names, key="scene")
    scene_from, scene_to = scenes[scene_name]
    if st.session_state.get("scene_shown") != scene_name:
        st.session_state.scene_shown = scene_name
        start = scene_from
        if "at" in st.query_params and "query_used" not in st.session_state:
            start = min(max(pd.Timestamp(st.query_params["at"]).floor(f"{STEP_MINUTES}min"), scene_from), scene_to)
            st.session_state.query_used = True
        st.session_state.cursor = start.to_pydatetime()
        st.session_state.playing = False
    playing = st.session_state.playing
    controls[1].button("Pause" if playing else "Play", icon=":material/pause:" if playing else ":material/play_arrow:", type="primary",
                       on_click=toggle_play, kwargs={"scene_from": scene_from, "scene_to": scene_to}, width="stretch")
    speed = controls[2].selectbox("Speed", list(SPEEDS), index=1, key="speed")
    show_zones = controls[3].toggle("Show the reported leaks", value=True, key="show_zones")
    zones = zone_rows(leaks=leaks, scene_from=scene_from, scene_to=scene_to)

    @st.fragment(run_every=PLAY_SECONDS if playing else None)
    def live_view():
        if st.session_state.playing:
            following = pd.Timestamp(st.session_state.cursor) + pd.Timedelta(minutes=STEP_MINUTES * SPEEDS[speed])
            if following >= scene_to:
                st.session_state.cursor = scene_to.to_pydatetime()
                st.session_state.playing = False
                st.rerun()
            st.session_state.cursor = following.to_pydatetime()
        st.slider("Time", min_value=scene_from.to_pydatetime(), max_value=scene_to.to_pydatetime(), step=timedelta(minutes=STEP_MINUTES),
                  format="D MMM YYYY, HH:mm", key="cursor", label_visibility="collapsed")
        cursor = pd.Timestamp(st.session_state.cursor)
        found = test.index[test["end"] == cursor]
        window = test.loc[found[0]] if len(found) else None

        status = st.columns([1.15, 1, 1])
        with status[0].container(border=True, height="stretch"):
            color, note = zone_note(cursor=cursor, leaks=leaks) if show_zones else ("#C9CBC6", "Reported leaks are hidden")
            data_note = "" if window is not None else ('<div class="lw-note"><span class="lw-dot" style="background:#5F5E5A"></span>'
                                                       'No window: the train was off or data is missing</div>')
            st.markdown(f'<div class="lw-label">Replay time</div><div class="lw-clock">{cursor:%a %d %b %Y, %H:%M}</div>'
                        f'<div class="lw-note"><span class="lw-dot" style="background:{color}"></span>{note}</div>{data_note}', unsafe_allow_html=True)
        for column, model in zip(status[1:], MODEL_NAMES):
            with column.container(border=True, height="stretch"):
                st.markdown(lamp_card(model=model, window=window, limit=limits[model], events=events, cursor=cursor), unsafe_allow_html=True)

        with st.container(border=True, gap=None):
            st.markdown("**What the compressor did**")
            for lane in compressor_lanes(readings=readings, zones=zones, scene_from=scene_from, scene_to=scene_to, cursor=cursor, show_zones=show_zones):
                st.altair_chart(lane, width="stretch")
        with st.container(border=True, gap=None):
            st.markdown(f"**What the models saw** &nbsp;<span style='color:{MUTED}'>dashed line: the limit, red dots: alarms</span>", unsafe_allow_html=True)
            for lane in model_lanes(test=test, zones=zones, limits=limits, scene_from=scene_from, scene_to=scene_to, cursor=cursor, show_zones=show_zones):
                st.altair_chart(lane, width="stretch")

        lower = st.columns([0.85, 1.15])
        with lower[0].container(border=True, height="stretch"):
            st.markdown("**Why this window looks unusual**")
            if window is None:
                st.caption("No window at this time.")
            else:
                lstm_parts = pd.Series({column.replace("lstm_error_", ""): window[column] for column in test.columns if column.startswith("lstm_error_")})
                forest_parts = parts.loc[parts["end"] == cursor].drop(columns="end").iloc[0]
                for model, model_parts in [("lstm_autoencoder", lstm_parts), ("isolation_forest", forest_parts)]:
                    st.markdown(f'<div class="lw-model" style="margin:0.4rem 0 0.35rem 0"><span class="lw-dot" style="background:{MODEL_COLORS[model]}"></span>'
                                f'{MODEL_NAMES[model]}</div>{why_rows(parts=model_parts, color=MODEL_COLORS[model])}', unsafe_allow_html=True)
                st.caption("Share of the LSTM error, and of the forest's push towards unusual (SHAP), per signal.")
        with lower[1].container(border=True, height="stretch"):
            st.markdown("**Alarm events so far in this moment**")
            started = events[(events["first_alarm"] >= scene_from) & (events["first_alarm"] <= cursor)].sort_values("first_alarm")
            if started.empty:
                st.caption("No alarm yet.")
            else:
                counted = started.apply(lambda row: f"leak {row['leak']}" if row["kind"] == "leak" else row["kind"], axis=1) if show_zones else "hidden"
                st.dataframe(pd.DataFrame({"started": started["first_alarm"].dt.strftime("%d %b %H:%M"), "model": started["model"],
                                           "counted as": counted, "looks like": started["looks_like"]}), hide_index=True, width="stretch")

    live_view()


def lead_text(hours):
    if pd.isna(hours):
        return "missed"
    if hours > 0:
        return f"{hours:.1f} h before the start"
    if hours == 0:
        return "at the start"
    return f"{-hours:.1f} h after the start"


def results_page():
    data = load_data()
    summary, leaks, events, tradeoff = data["summary"], data["leaks"], data["events"], data["tradeoff"]
    page_header(title="Results", lead="Both models learned from February only. Their limits were locked on March. Then the test months, April to August 2020, were opened once.")
    lead_columns = [f"lead time leak {leak} (h)" for leak in leaks["leak"]]
    cards = st.columns(2)
    for column, model in zip(cards, MODEL_NAMES):
        row = summary.loc[MODEL_NAMES[model]]
        early = [(leak, row[name]) for leak, name in zip(leaks["leak"], lead_columns) if row[name] > 0]
        early_text = ", ".join(f"leak {leak}, {hours:.1f} h before" for leak, hours in early) if early else "none"
        weeks = row["false alarm events"] / row["false alarms per week"]
        with column.container(border=True):
            st.markdown(f'<div class="lw-model"><span class="lw-dot" style="background:{MODEL_COLORS[model]}"></span>{MODEL_NAMES[model]}</div>', unsafe_allow_html=True)
            numbers = st.columns(3)
            numbers[0].markdown(f'<div class="lw-label">leaks caught</div><div class="lw-big">{int(row["leaks caught"])} of {len(leaks)}</div>', unsafe_allow_html=True)
            numbers[1].markdown(f'<div class="lw-label">warned before the start</div><div class="lw-big">{len(early)}</div><div class="lw-since">{early_text}</div>',
                                unsafe_allow_html=True)
            numbers[2].markdown(f'<div class="lw-label">false alarms per week</div><div class="lw-big">{row["false alarms per week"]:.2f}</div>'
                                f'<div class="lw-since">{int(row["false alarm events"])} in {weeks:.1f} healthy weeks</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**The first alarm for every leak**")
        table = pd.DataFrame({"leak": leaks["leak"], "reported start": leaks["start"].dt.strftime("%d %b %Y %H:%M"),
                              "length (h)": ((leaks["end"] - leaks["start"]) / pd.Timedelta(hours=1)).round(1)})
        for model in MODEL_NAMES:
            table[MODEL_NAMES[model]] = [lead_text(hours=summary.loc[MODEL_NAMES[model], name]) for name in lead_columns]
        st.dataframe(table, hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown(f"**Every alarm event of the test months** &nbsp;<span style='color:{MUTED}'>bands: the reported leaks with their warning and ignored zones</span>",
                    unsafe_allow_html=True)
        test = data["test"]
        test_from, test_to = test["end"].min().floor("D"), test["end"].max().ceil("D")
        zones = in_utc(frame=in_utc(frame=zone_rows(leaks=leaks, scene_from=test_from, scene_to=test_to), column="start"), column="end")
        frame = events.assign(end=events["last_alarm"] + pd.Timedelta(hours=6),
                              counted=events["kind"].map({"leak": "during a leak", "false alarm": "false alarm", "ignored zone": "ignored zone"}))
        frame = in_utc(frame=in_utc(frame=frame, column="first_alarm"), column="end")
        x = alt.X("first_alarm:T", scale=alt.Scale(type="utc", domain=[vega_time(stamp=test_from), vega_time(stamp=test_to)]),
                  axis=alt.Axis(title=None, format="%b", tickCount="month", grid=False, labelColor=MUTED))
        bands = alt.Chart(zones).mark_rect(opacity=0.22).encode(x=alt.X("start:T", scale=alt.Scale(type="utc")), x2="end:T",
                                                                  color=alt.Color("zone:N", scale=alt.Scale(domain=list(ZONE_COLORS), range=list(ZONE_COLORS.values())), legend=None))
        bars = alt.Chart(frame).mark_bar(height=16, cornerRadius=2).encode(
            x=x, x2="end:T", y=alt.Y("model:N", title=None, sort=list(MODEL_NAMES.values()), axis=alt.Axis(labelColor=INK, domain=False, ticks=False)),
            color=alt.Color("counted:N", title=None, scale=alt.Scale(domain=["during a leak", "false alarm", "ignored zone"], range=[INK, ALARM_COLOR, "#B4B2A9"]),
                            legend=alt.Legend(orient="top", direction="horizontal")),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("first_alarm:T", title="first alarm", format="%d %b %H:%M", formatType="utc"),
                     alt.Tooltip("counted:N", title="counted as"), alt.Tooltip("looks_like:N", title="looks like")])
        st.altair_chart(alt.layer(bands, bars).resolve_scale(color="independent").properties(height=150).configure_view(stroke=None), width="stretch")
        st.caption("Each bar is one alarm event, drawn a little wider so that short events stay visible. Hover over a bar to see what it looks like.")

    with st.expander("What other limits would have given (a what-if after the test, notebook 5.4)"):
        labels = list(dict.fromkeys(tradeoff["setting"]))
        long = tradeoff.melt(id_vars=["model", "setting"], value_vars=["false alarms per week", "longest early warning (h)"], var_name="measure")
        st.markdown(" &nbsp; ".join(f'<span class="lw-dot" style="background:{MODEL_COLORS[model]}"></span> {MODEL_NAMES[model]}' for model in MODEL_NAMES),
                    unsafe_allow_html=True)
        for column, (measure, title) in zip(st.columns(2), [("false alarms per week", "The price: false alarms per week in the test months"),
                                                            ("longest early warning (h)", "The gain: longest early warning (hours before a leak start)")]):
            chart = alt.Chart(long[long["measure"] == measure]).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("setting:N", sort=labels, title="budget in March (false alarm events per week)", axis=alt.Axis(labelAngle=0, labelColor=MUTED, titleColor=MUTED)),
                xOffset=alt.XOffset("model:N", sort=list(MODEL_NAMES.values())), y=alt.Y("value:Q", title=None, axis=alt.Axis(labelColor=MUTED, tickCount=4)),
                color=alt.Color("model:N", scale=alt.Scale(domain=list(MODEL_NAMES.values()), range=list(MODEL_COLORS.values())), legend=None),
                tooltip=["model", "setting", alt.Tooltip("value:Q", format=".2f")])
            column.markdown(f"**{title}**")
            column.altair_chart(chart.properties(height=220).configure_view(stroke=None), width="stretch")
        st.caption("Each budget gave its own limit on March. The official results stay those of the locked budget (1). For the LSTM, looser budgets only add false alarms; "
                   "the forest stays squeezed under the ceiling of its score.")

    with st.container(border=True):
        st.markdown("**Good to know**")
        st.markdown("- **Healthy only means that no leak was reported.** Some false alarms look just like the reported leaks.\n"
                    "- **Frozen readings.** For some hours of the test months, the readings did not move at all. One long false alarm of the LSTM comes from them.\n"
                    "- **Only 4 leaks.** Each lead time rests on a single leak.")


def how_page():
    page_header(title="How it works", lead="From the readings of one compressor to an alarm, in four steps.")
    with st.container(border=True):
        st.image(AIR_SYSTEM_IMAGE, caption="The air system of the train, with the column that measures each part", width="stretch")
    steps = st.columns(4)
    for column, (number, title, text) in zip(steps, [
        ("1", "Readings", "15 signals of one metro train compressor, about every 10 seconds, from February to August 2020 (MetroPT-3)."),
        ("2", "Windows", "Every 5 minutes, the last 60 minutes of readings become one window."),
        ("3", "Two models, trained on February only", "Isolation Forest scores 40 summary numbers of the window: how easy it is to cut away from normal windows. "
                                                    "The LSTM autoencoder tries to redraw the 360 ticks of 12 signals: the worse it redraws them, the higher its error."),
        ("4", "Alarms", "An alarm needs 3 windows in a row above the limit, and alarms at most 60 minutes apart are one event. "
                       "Each limit was locked on March: the lowest one with at most 1 false alarm per week."),
    ]):
        with column.container(border=True, height="stretch"):
            st.markdown(f'<div class="lw-step">Step {number}</div>', unsafe_allow_html=True)
            st.markdown(f"**{title}**  \n{text}")
    st.caption("This app replays saved results: no model runs here. Every score comes from the notebooks.")


check_files()
pages = [st.Page(replay_page, title="Replay", icon=":material/replay:", default=True),
         st.Page(results_page, title="Results", icon=":material/table_chart:", url_path="results"),
         st.Page(how_page, title="How it works", icon=":material/schema:", url_path="how-it-works")]
st.navigation(pages, position="top").run()
