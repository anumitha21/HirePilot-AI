# HirePilot AI

Voice-first adaptive AI interview agent, built one milestone at a time.

## M0: Scaffolding + Raw Voice Loop

This milestone sets up the backend scaffold, environment configuration, logging, and a raw local voice loop:

```text
microphone -> faster-whisper -> echo text -> Edge TTS -> speaker
```

No AI interview logic is included yet.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

## Run M0

```powershell
python -m backend.scripts.raw_voice_loop
```

Speak after the prompt. The app records a short clip, transcribes it, then speaks the text back.

To test speaker output only:

```powershell
python -m backend.scripts.test_speaker
```

The voice loop uses simple VAD-style recording: it keeps recording until you stop speaking, up to `VOICE_MAX_RECORD_SECONDS`.

## M1: Resume & JD Understanding

M1 extracts structured resume and job-description JSON using Pydantic schemas. Normal mode uses Groq JSON output; sample mode verifies the contract locally.

```powershell
python -m backend.scripts.understand_resume_jd --sample --local
```

To use Groq, set `GROQ_API_KEY` in `.env` and run:

```powershell
python -m backend.scripts.understand_resume_jd --resume-file backend\samples\sample_resume.txt --jd-file backend\samples\sample_jd.txt
```

## M2: Planner Agent

M2 turns structured resume/JD understanding into an interview plan. Local mode verifies the schema contract without calling Groq.

```powershell
python -m backend.scripts.plan_interview --sample --local
```

To use Groq, set `GROQ_API_KEY` in `.env` and run:

```powershell
python -m backend.scripts.plan_interview --sample
```

## M3: Shared InterviewState + Retriever

M3 adds a shared interview state object and a FAISS-backed retriever for resume, job-description, and interview-guideline context.

```powershell
python -m backend.scripts.query_retriever --sample --query "FastAPI API design"
```

## M4: Interview Agent - Live Voice Loop

M4 adds adaptive question generation over the existing voice pipeline. The local text mode verifies 2-3 adaptive turns without a microphone or Groq key.

```powershell
python -m backend.scripts.live_interview --sample --local --text
```

To run the live voice loop with Groq, set `GROQ_API_KEY` in `.env` and run:

```powershell
python -m backend.scripts.live_interview --sample
```

## M5: Evaluation Agent

M5 scores a candidate answer across rubric dimensions and writes the result into `InterviewState.current_scores`.

```powershell
python -m backend.scripts.evaluate_answer --sample --local
```
