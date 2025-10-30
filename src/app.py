import streamlit as st
import os
import sys # NEW: Import sys
import json
import pandas as pd
import tempfile
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.stt import transcribe_audio
from src.diarize import perform_diarization, align_transcription_with_diarization
from src.extractor import extract_action_items, summarize_meeting

st.set_page_config(layout="wide", page_title="NotesAI | Smart Meeting Notes & Action-Item Extractor")

st.title("NotesAI")
st.markdown("Upload your meeting audio to get a timestamped transcript, speaker diarization, a concise summary, and a list of action items.")

uploaded_file = st.file_uploader("Upload an audio file (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    st.audio(audio_path, format=f"audio/{Path(uploaded_file.name).suffix[1:]}")
    st.success(f"File uploaded successfully: {uploaded_file.name}")

    if st.button("Process Audio", help="Click to start transcription, diarization, and extraction"):
        with st.spinner("Processing audio... This might take a few minutes for longer audio files."):
            try:
                # --- Step 1: Transcription ---
                st.subheader("1. Transcription")
                whisper_result = transcribe_audio(audio_path)
                whisper_segments = whisper_result["segments"]
                st.info(f"Transcription complete. Found {len(whisper_segments)} segments.")

                # --- Step 2: Diarization ---
                st.subheader("2. Speaker Diarization")
                diarization_segments = perform_diarization(audio_path)
                st.info(f"Diarization complete. Found {len(diarization_segments)} speaker turns.")

                # --- Step 3: Alignment ---
                st.subheader("3. Aligned Transcript")
                speaker_tagged_transcript = align_transcription_with_diarization(
                    whisper_segments, diarization_segments
                )

                # Format the full transcript for display and download
                full_transcript_text = []
                for segment in speaker_tagged_transcript:
                    full_transcript_text.append(
                        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}"
                    )
                st.text_area("Full Speaker-Tagged Transcript", "\n".join(full_transcript_text), height=300)

                # --- Step 4: Extraction ---
                st.subheader("4. Extracted Insights")

                # Action Items
                st.markdown("### Action Items")
                action_items = extract_action_items(speaker_tagged_transcript)
                if action_items:
                    # Convert action items to a DataFrame for better display
                    actions_df = pd.DataFrame([
                        {"Action": item['action'], "Owner": item['owner'], "Due Date": item['due_date']}
                        for item in action_items
                    ])
                    st.dataframe(actions_df)

                    # Summary
                    st.markdown("### Meeting Summary")
                    summary_bullets = summarize_meeting(speaker_tagged_transcript, num_bullets=5)
                    for bullet in summary_bullets:
                        st.markdown(bullet)
                else:
                    st.info("No action items or summary extracted.")

                # --- Downloadable Artifacts ---
                st.subheader("Downloadable Artifacts")

                # Transcript.txt
                st.download_button(
                    label="Download Transcript (TXT)",
                    data="\n".join(full_transcript_text),
                    file_name="transcript.txt",
                    mime="text/plain",
                    help="Download the full speaker-tagged transcript."
                )

                # Actions.json
                actions_json = json.dumps([
                    {"action": item['action'], "owner": item['owner'], "due_date": item['due_date']}
                    for item in action_items
                ], indent=2)
                st.download_button(
                    label="Download Action Items (JSON)",
                    data=actions_json,
                    file_name="actions.json",
                    mime="application/json",
                    help="Download the extracted action items in JSON format."
                )

                # Summary.md
                summary_md = "\n".join(summary_bullets)
                st.download_button(
                    label="Download Summary (Markdown)",
                    data=summary_md,
                    file_name="summary.md",
                    mime="text/markdown",
                    help="Download the meeting summary in Markdown format."
                )

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up the temporary audio file
                os.unlink(audio_path)