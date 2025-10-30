# NotesAI: Smart Meeting Notes & Action-Item Extractor

This project aims to convert meeting audio into a clean transcript, a concise meeting summary, and a prioritized list of action items with owners and due dates. The output can be exported as JSON/Markdown and viewed in a lightweight web demonstration.

## Core Deliverables (MVP)

1.  **Transcription** (timestamped) from uploaded audio (mp3/wav).
2.  **Speaker Diarization** (who said what; optional fallback: labelled Speaker 1/2).
3.  **Concise Meeting Summary** (3–5 bullets).
4.  **Action-Items List**: action text, proposed owner, proposed due date/confidence score.
5.  **Downloadable Artifacts**: `transcript.txt`, `actions.json`, `summary.md`.
6.  Lightweight web demo (Streamlit) + public GitHub repository with README and 3-minute demo GIF.

## Technology Stack

*   **STT:** Whisper (whisper.cpp or open-source Whisper)
*   **Diarization:** pyannote.audio (or simple speaker-turn heuristics)
*   **NLP Extraction:** Rule-based (spaCy, regex) + instruction-tuned open-source LLM (quantized LLaMA/GGML via llama.cpp or small HF model)
*   **Storage / Search:** Local JSON + simple file store
*   **Frontend:** Streamlit
*   **Optional:** WhisperX for better timestamps and alignment

## One-Week Timeline

*   **Day 1: Scope & Infrastructure:** Freeze MVP scope. Spin up repository template, Dockerfile, example audio dataset, and dev environment.
*   **Day 2: Transcription:** Integrate Whisper. Output timestamped transcript.
*   **Day 3: Diarization + Alignment:** Add pyannote or simple speaker-split. Align speakers to timestamps. Produce speaker-tagged transcript.
*   **Day 4: Extraction:** Implement extractor: rules for verbs, NER for names/dates, LLM for disambiguation.
*   **Day 5: UI + Download:** Streamlit app: upload → run pipeline → show transcript, summary, actions; export JSON/MD.
*   **Day 6: Polish + Tests:** Add unit tests for extractor, sample audio tests, README, 3-minute screencast GIF.
*   **Day 7: Deploy & Write:** Deploy demo, finalize repository, short blog post.
