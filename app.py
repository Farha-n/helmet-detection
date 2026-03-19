from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import streamlit as st
from ultralytics import YOLO

from reporting import generate_report

VIOLATION_SAVE_COOLDOWN_SECONDS = 5
MAX_RECENT_EVENTS = 20


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_label_map(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def parse_helmet_labels(raw_labels: str) -> set[str]:
    return {label.strip().lower() for label in raw_labels.split(",") if label.strip()}


def is_helmet_on_head(person_box: tuple[int, int, int, int], helmet_boxes: list[tuple[int, int, int, int]]) -> bool:
    px1, py1, px2, py2 = person_box
    head_y2 = py1 + int((py2 - py1) * 0.35)

    for hx1, hy1, hx2, hy2 in helmet_boxes:
        cx = (hx1 + hx2) // 2
        cy = (hy1 + hy2) // 2
        inside_person = px1 <= cx <= px2 and py1 <= cy <= py2
        inside_head_band = py1 <= cy <= head_y2
        if inside_person and inside_head_band:
            return True
    return False


def append_report(log_file: Path, payload: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "last_status": "",
        "last_saved_at": dt.datetime.min,
        "latest_status": "UNKNOWN",
        "latest_report_summary": "No report generated yet.",
        "total_reports_generated": 0,
        "total_violation_events": 0,
        "saved_violation_frames": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "recent_events" not in st.session_state:
        st.session_state["recent_events"] = []


def reset_session_metrics() -> None:
    st.session_state["last_status"] = ""
    st.session_state["last_saved_at"] = dt.datetime.min
    st.session_state["latest_status"] = "UNKNOWN"
    st.session_state["latest_report_summary"] = "No report generated yet."
    st.session_state["total_reports_generated"] = 0
    st.session_state["total_violation_events"] = 0
    st.session_state["saved_violation_frames"] = 0
    st.session_state["recent_events"] = []


def status_label(status: str) -> str:
    if status == "NO HELMET":
        return "Violation: No Helmet"
    if status == "HELMET":
        return "Helmet Compliant"
    if status == "SAFE":
        return "Safe"
    return "Unknown"


def render_status_panel(container: Any, status: str, supports_helmet_classes: bool) -> None:
    mode_text = "Helmet model mode" if supports_helmet_classes else "Fallback mode (no helmet class in model)"
    message = f"{status_label(status)} | {mode_text}"

    if status == "NO HELMET":
        container.error(message)
    elif status in ("HELMET", "SAFE"):
        container.success(message)
    else:
        container.info(message)


def render_live_metrics(container: Any, stats: dict[str, Any]) -> None:
    with container.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", stats["status"])
        col2.metric("People", stats["person_count"])
        col3.metric("Violations", stats["violation_count"])


def render_sidebar_metrics(container: Any, current_stats: dict[str, Any] | None = None) -> None:
    with container.container():
        st.subheader("System Metrics")

        col1, col2 = st.columns(2)
        col1.metric("Violation events", st.session_state["total_violation_events"])
        col2.metric("Saved frames", st.session_state["saved_violation_frames"])

        col3, col4 = st.columns(2)
        col3.metric("Reports", st.session_state["total_reports_generated"])
        col4.metric("Latest status", st.session_state["latest_status"])

        if current_stats is not None:
            st.caption(
                f"Live frame: people={current_stats['person_count']}, violations={current_stats['violation_count']}"
            )


def add_recent_event(report: dict[str, Any], saved_image_path: str | None) -> None:
    event = {
        "time": report.get("timestamp", "-"),
        "status": report.get("status", "-"),
        "violations": report.get("violation_count", 0),
        "saved_image": saved_image_path or "-",
    }

    events: list[dict[str, Any]] = st.session_state["recent_events"]
    events.insert(0, event)
    st.session_state["recent_events"] = events[:MAX_RECENT_EVENTS]


def render_recent_events(container: Any) -> None:
    with container.container():
        st.subheader("Recent Events")
        events: list[dict[str, Any]] = st.session_state["recent_events"]
        if not events:
            st.info("No events recorded yet.")
            return
        st.dataframe(events, use_container_width=True, hide_index=True)


def read_latest_report(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None

    try:
        with log_file.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
    except OSError:
        return None

    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def save_violation_frame(frame: Any, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    file_name = f"violation_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    out_path = output_dir / file_name
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def analyze_frame(
    frame: Any,
    model: YOLO,
    label_map: dict[int, str],
    confidence: float,
    helmet_labels: set[str],
) -> tuple[Any, dict[str, Any]]:
    results = model.predict(frame, conf=confidence, verbose=False)

    person_boxes: list[tuple[int, int, int, int]] = []
    helmet_boxes: list[tuple[int, int, int, int]] = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = label_map.get(cls_id, str(cls_id)).lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if label == "person":
                person_boxes.append((x1, y1, x2, y2))
            elif label in helmet_labels:
                helmet_boxes.append((x1, y1, x2, y2))

    available_labels = {name.lower() for name in label_map.values()}
    supports_helmet_classes = len(helmet_labels.intersection(available_labels)) > 0

    violations = 0
    for person_box in person_boxes:
        has_helmet = is_helmet_on_head(person_box, helmet_boxes) if supports_helmet_classes else False
        if not has_helmet:
            violations += 1

        x1, y1, x2, y2 = person_box
        color = (0, 220, 0) if has_helmet else (0, 0, 255)
        label = "Helmet" if has_helmet else "No Helmet"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for hx1, hy1, hx2, hy2 in helmet_boxes:
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 180, 0), 2)

    person_count = len(person_boxes)
    if person_count == 0:
        status = "SAFE"
        status_color = (0, 220, 0)
    elif violations > 0:
        status = "NO HELMET"
        status_color = (0, 0, 255)
    else:
        status = "HELMET"
        status_color = (0, 220, 0)

    mode_text = "Mode: helmet model" if supports_helmet_classes else "Mode: fallback (no helmet class)"
    cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(
        frame,
        f"People: {person_count} | Violations: {violations}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, mode_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame, {
        "status": status,
        "person_count": person_count,
        "violation_count": violations,
        "supports_helmet_classes": supports_helmet_classes,
    }


def maybe_report_event(
    stats: dict[str, Any],
    source: str,
    log_file: Path,
    save_violations: bool,
    output_dir: Path,
    frame: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    last_status = st.session_state.get("last_status", "")
    last_saved_at = st.session_state.get("last_saved_at", dt.datetime.min)

    now = dt.datetime.now()
    saved_image_path = None
    if save_violations and stats["violation_count"] > 0 and (now - last_saved_at).total_seconds() >= VIOLATION_SAVE_COOLDOWN_SECONDS:
        saved_image_path = save_violation_frame(frame, output_dir)
        st.session_state["last_saved_at"] = now
        st.session_state["saved_violation_frames"] += 1

    should_log = stats["status"] != last_status or saved_image_path is not None
    if not should_log:
        return None, saved_image_path

    report = generate_report(
        status=stats["status"],
        person_count=stats["person_count"],
        violation_count=stats["violation_count"],
        extra={
            "source": source,
            "supports_helmet_classes": stats["supports_helmet_classes"],
            "saved_image_path": saved_image_path,
        },
    )
    append_report(log_file, report)
    st.session_state["last_status"] = stats["status"]
    st.session_state["latest_status"] = stats["status"]
    st.session_state["latest_report_summary"] = report["summary"]
    st.session_state["total_reports_generated"] += 1
    if stats["violation_count"] > 0:
        st.session_state["total_violation_events"] += 1
    add_recent_event(report, saved_image_path)

    return report, saved_image_path


def run_video_mode(
    uploaded_file: Any,
    model: YOLO,
    label_map: dict[int, str],
    confidence: float,
    helmet_labels: set[str],
    save_violations: bool,
    output_dir: Path,
    log_file: Path,
    sidebar_metrics_placeholder: Any,
) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = Path(tmp_file.name)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return

    status_placeholder = st.empty()
    live_metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    report_alert_placeholder = st.empty()
    report_payload_placeholder = st.empty()
    saved_image_placeholder = st.empty()
    events_placeholder = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0.0)

    st.subheader("Live Detection Feed")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, stats = analyze_frame(frame, model, label_map, confidence, helmet_labels)
        report, saved_image_path = maybe_report_event(
            stats=stats,
            source=f"upload:{uploaded_file.name}",
            log_file=log_file,
            save_violations=save_violations,
            output_dir=output_dir,
            frame=annotated,
        )

        render_status_panel(status_placeholder, stats["status"], stats["supports_helmet_classes"])
        render_live_metrics(live_metrics_placeholder, stats)
        render_sidebar_metrics(sidebar_metrics_placeholder, stats)
        render_recent_events(events_placeholder)

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        if report:
            if stats["violation_count"] > 0:
                report_alert_placeholder.error(f"AI Report: {report['summary']}")
            else:
                report_alert_placeholder.info(f"AI Report: {report['summary']}")
            report_payload_placeholder.json(report)
        if saved_image_path:
            saved_image_placeholder.caption(f"Saved violation frame: {saved_image_path}")

        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    tmp_path.unlink(missing_ok=True)
    render_sidebar_metrics(sidebar_metrics_placeholder)
    st.success("Video processing completed.")


def run_webcam_mode(
    camera_index: int,
    max_frames: int,
    model: YOLO,
    label_map: dict[int, str],
    confidence: float,
    helmet_labels: set[str],
    save_violations: bool,
    output_dir: Path,
    log_file: Path,
    sidebar_metrics_placeholder: Any,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Could not open webcam. Check camera index and permissions.")
        return

    status_placeholder = st.empty()
    live_metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    report_alert_placeholder = st.empty()
    report_payload_placeholder = st.empty()
    saved_image_placeholder = st.empty()
    events_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    st.subheader("Live Detection Feed")

    for idx in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        annotated, stats = analyze_frame(frame, model, label_map, confidence, helmet_labels)
        report, saved_image_path = maybe_report_event(
            stats=stats,
            source=f"webcam:{camera_index}",
            log_file=log_file,
            save_violations=save_violations,
            output_dir=output_dir,
            frame=annotated,
        )

        render_status_panel(status_placeholder, stats["status"], stats["supports_helmet_classes"])
        render_live_metrics(live_metrics_placeholder, stats)
        render_sidebar_metrics(sidebar_metrics_placeholder, stats)
        render_recent_events(events_placeholder)

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        if report:
            if stats["violation_count"] > 0:
                report_alert_placeholder.error(f"AI Report: {report['summary']}")
            else:
                report_alert_placeholder.info(f"AI Report: {report['summary']}")
            report_payload_placeholder.json(report)
        if saved_image_path:
            saved_image_placeholder.caption(f"Saved violation frame: {saved_image_path}")

        progress_bar.progress((idx + 1) / max_frames)

    cap.release()
    render_sidebar_metrics(sidebar_metrics_placeholder)
    st.success("Webcam batch completed. Run again for continued monitoring.")


def main() -> None:
    initialize_session_state()

    st.set_page_config(page_title="AI-Based Industrial Safety Monitoring System", layout="wide")

    st.title("AI-Based Industrial Safety Monitoring System")
    st.write("Web-based helmet compliance monitoring with upload and webcam support.")

    with st.sidebar:
        st.title("Controls")
        st.header("Detection Settings")
        model_path = st.text_input("YOLO model path", value="yolov8n.pt")
        confidence = st.slider("Confidence", min_value=0.1, max_value=0.95, value=0.35, step=0.05)
        helmet_labels_raw = st.text_input("Helmet labels", value="helmet,hardhat")

        st.header("Violation Settings")
        save_violations = st.checkbox("Save violation frames", value=True)
        output_dir = Path(st.text_input("Violation folder", value="violations"))
        report_file = Path(st.text_input("Report log file", value="reports/safety_log.jsonl"))

        if st.button("Reset session metrics"):
            reset_session_metrics()
            st.success("Session metrics reset.")

        sidebar_metrics_placeholder = st.empty()

    render_sidebar_metrics(sidebar_metrics_placeholder)

    try:
        model = load_model(model_path)
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        return

    label_map = get_label_map(model)
    helmet_labels = parse_helmet_labels(helmet_labels_raw)

    mode = st.radio("Input source", options=["Upload Video", "Webcam"], horizontal=True)

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None and st.button("Run Detection", type="primary"):
            run_video_mode(
                uploaded_file=uploaded_file,
                model=model,
                label_map=label_map,
                confidence=confidence,
                helmet_labels=helmet_labels,
                save_violations=save_violations,
                output_dir=output_dir,
                log_file=report_file,
                sidebar_metrics_placeholder=sidebar_metrics_placeholder,
            )

    if mode == "Webcam":
        camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        max_frames = st.slider("Frames per run", min_value=50, max_value=1000, value=300, step=50)
        if st.button("Run Webcam", type="primary"):
            run_webcam_mode(
                camera_index=int(camera_index),
                max_frames=max_frames,
                model=model,
                label_map=label_map,
                confidence=confidence,
                helmet_labels=helmet_labels,
                save_violations=save_violations,
                output_dir=output_dir,
                log_file=report_file,
                sidebar_metrics_placeholder=sidebar_metrics_placeholder,
            )

    st.divider()
    st.subheader("Automated AI Reporting")
    if st.session_state["latest_status"] == "NO HELMET":
        st.error(st.session_state["latest_report_summary"])
    else:
        st.info(st.session_state["latest_report_summary"])

    st.divider()
    st.subheader("Latest Safety Report")
    if st.button("Refresh Latest Report"):
        report = read_latest_report(report_file)
        if report is None:
            st.info("No report has been generated yet.")
        else:
            st.json(report)


if __name__ == "__main__":
    main()
