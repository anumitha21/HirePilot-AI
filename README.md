# HirePilot AI

HirePilot AI is a voice-first adaptive interview agent that turns a resume and job description into a live, recruiter-ready interview experience. The project now includes a polished web demo, a structured interview graph, report generation, and local persistence for demoing.

## What is included

- Adaptive interview flow with planner, retriever, interviewer, evaluator, and report nodes
- Local deterministic agents for demos when no Groq key is available
- A simple polished UI for entering resume/JD details and viewing a generated report
- JSON-backed interview storage for recent demo runs

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest fastapi uvicorn
```

### Run the polished demo UI

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

### Run the backend interview graph locally

```powershell
python -m backend.scripts.run_interview_graph --sample --local --text
```

### Run the regression tests

```powershell
python -m pytest backend/tests/test_report_and_persistence.py
```

## Demo flow

1. Open the web UI.
2. Enter a candidate name and short resume/JD summary.
3. Click Start Demo Interview.
4. The backend runs the graph, generates a score, and stores a recruiter-style report.
