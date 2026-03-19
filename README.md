# AI Helmet Detection System

A practical, interview-ready AI safety monitoring project built with YOLO, OpenCV, and Flask.

## What this project does

- Reads live webcam/video stream
- Detects people with YOLO
- Flags safety state as:
  - HELMET
  - NO HELMET
  - SAFE (no person)
- Saves violation snapshots (optional)
- Generates machine-readable safety reports
- Exposes Flask API endpoints for status/report access

## Important model note

Default YOLO weights like `yolov8n.pt` detect person but usually do not include a helmet class.

This project handles that in two modes:

- Helmet model mode: if your model includes helmet labels (for example `helmet` or `hardhat`), per-person helmet logic is used.
- Fallback mode: if helmet labels are not present, each detected person is marked as `NO HELMET` to simulate a safety violation workflow.

## Tech stack

- Python 3.9-3.11
- OpenCV
- Ultralytics YOLO
- Flask
- Optional OpenAI API integration for AI summaries

## Quick start (Windows)

1. Create and activate virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run detection app:

```powershell
python main.py --source 0 --save-violations
```

4. Press `ESC` or `q` to exit camera view.

## Flask API

Run API:

```powershell
python app.py
```

Available endpoints:

- `GET /` -> service metadata
- `GET /health` -> health check
- `GET /latest-report` -> most recent report from log file
- `POST /generate-report` -> manually generate and persist a report

Example request:

```bash
curl -X POST http://127.0.0.1:5000/generate-report \
  -H "Content-Type: application/json" \
  -d "{\"status\":\"NO HELMET\",\"person_count\":2,\"violation_count\":1}"
```

## OpenAI-powered report summaries (optional)

Set environment variables before running:

```powershell
$env:ENABLE_LLM_REPORTS="1"
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

When enabled, each generated report attempts to include an `ai_summary` field.

## Suggested next upgrade

- Train custom YOLO helmet model on Roboflow dataset
- Replace fallback logic with true helmet detections
- Build React dashboard for live alerts and historical analytics
