from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from reporting import generate_report

VIOLATION_SAVE_COOLDOWN_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Helmet Detection System")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video file path (default: 0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--helmet-labels",
        default="helmet,hardhat",
        help="Comma-separated class names treated as helmet labels",
    )
    parser.add_argument(
        "--save-violations",
        action="store_true",
        help="Save a frame when No Helmet is detected",
    )
    parser.add_argument(
        "--output-dir",
        default="violations",
        help="Directory for saved violation frames",
    )
    parser.add_argument(
        "--report-file",
        default="reports/safety_log.jsonl",
        help="JSONL file used for report logs",
    )
    return parser.parse_args()


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def get_label_map(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


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


def write_report(log_file: Path, payload: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    label_map = get_label_map(model)

    helmet_labels = {label.strip().lower() for label in args.helmet_labels.split(",") if label.strip()}
    available_labels = {name.lower() for name in label_map.values()}
    supports_helmet_classes = len(helmet_labels.intersection(available_labels)) > 0

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    output_dir = Path(args.output_dir)
    log_file = Path(args.report_file)

    print("[INFO] Running detection. Press ESC or q to exit.")
    if not supports_helmet_classes:
        print(
            "[WARN] Model has no helmet label. Running fallback mode: every detected person is marked as No Helmet."
        )

    last_saved_at = dt.datetime.min
    last_status = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Stream ended or frame read failed.")
            break

        results = model.predict(frame, conf=args.confidence, verbose=False)

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

        person_count = len(person_boxes)
        violations = 0

        for person_box in person_boxes:
            has_helmet = is_helmet_on_head(person_box, helmet_boxes) if supports_helmet_classes else False
            is_violation = not has_helmet
            if is_violation:
                violations += 1

            x1, y1, x2, y2 = person_box
            color = (0, 220, 0) if has_helmet else (0, 0, 255)
            label = "Helmet" if has_helmet else "No Helmet"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for hx1, hy1, hx2, hy2 in helmet_boxes:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 180, 0), 2)

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
        cv2.putText(frame, f"People: {person_count} | Violations: {violations}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, mode_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        saved_image_path = None
        now = dt.datetime.now()
        if args.save_violations and violations > 0 and (now - last_saved_at).total_seconds() >= VIOLATION_SAVE_COOLDOWN_SECONDS:
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            saved_path = output_dir / filename
            cv2.imwrite(str(saved_path), frame)
            saved_image_path = str(saved_path)
            last_saved_at = now

        should_log = status != last_status or saved_image_path is not None
        if should_log:
            report = generate_report(
                status=status,
                person_count=person_count,
                violation_count=violations,
                extra={
                    "source": str(args.source),
                    "supports_helmet_classes": supports_helmet_classes,
                    "saved_image_path": saved_image_path,
                },
            )
            write_report(log_file, report)
            print(report["summary"])
            last_status = status

        cv2.imshow("AI Safety System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
