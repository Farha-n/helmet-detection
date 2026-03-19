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
) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = Path(tmp_file.name)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return

    frame_placeholder = st.empty()
    report_placeholder = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0.0)

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", stats["status"])
        col2.metric("People", stats["person_count"])
        col3.metric("Violations", stats["violation_count"])

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        if report:
            report_placeholder.json(report)
        if saved_image_path:
            st.caption(f"Saved violation frame: {saved_image_path}")

        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    tmp_path.unlink(missing_ok=True)
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
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Could not open webcam. Check camera index and permissions.")
        return

    frame_placeholder = st.empty()
    report_placeholder = st.empty()
    progress_bar = st.progress(0.0)

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", stats["status"])
        col2.metric("People", stats["person_count"])
        col3.metric("Violations", stats["violation_count"])

        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        if report:
            report_placeholder.json(report)
        if saved_image_path:
            st.caption(f"Saved violation frame: {saved_image_path}")

        progress_bar.progress((idx + 1) / max_frames)

    cap.release()
    st.success("Webcam batch completed. Run again for continued monitoring.")


def main() -> None:
    st.set_page_config(page_title="AI-Based Industrial Safety Monitoring System", layout="wide")

    st.title("AI-Based Industrial Safety Monitoring System")
    st.write("Web-based helmet compliance monitoring with upload and webcam support.")

    with st.sidebar:
        st.header("Detection Settings")
        model_path = st.text_input("YOLO model path", value="yolov8n.pt")
        confidence = st.slider("Confidence", min_value=0.1, max_value=0.95, value=0.35, step=0.05)
        helmet_labels_raw = st.text_input("Helmet labels", value="helmet,hardhat")

        st.header("Violation Settings")
        save_violations = st.checkbox("Save violation frames", value=True)
        output_dir = Path(st.text_input("Violation folder", value="violations"))
        report_file = Path(st.text_input("Report log file", value="reports/safety_log.jsonl"))

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
            )

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
