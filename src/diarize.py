from .stt import transcribe_audio
from pyannote.audio import Pipeline
import os

HF_TOKEN = os.environ.get("HF_TOKEN")

def perform_diarization(audio_path: str) -> list:
    """
    Performs speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        list: A list of speaker segments (e.g., [(start_time, end_time, speaker_label), ...])
    """
    if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
        raise ValueError(
            "Hugging Face access token (HF_TOKEN) not found in environment variables. "
            "Please set the HF_TOKEN environment variable (e.g., $env:HF_TOKEN='hf_YOUR_TOKEN') "
            "and agree to user conditions for the model "
            "at https://huggingface.co/pyannote/speaker-diarization-3.1."
        )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    diarization = pipeline(audio_path)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((turn.start, turn.end, speaker))

    return speaker_segments

def align_transcription_with_diarization(
    whisper_segments: list[dict],
    diarization_segments: list[tuple]
) -> list[dict]:
    """
    Aligns Whisper's timestamped transcription segments with Pyannote.audio's speaker diarization segments.

    Args:
        Whisper_segments (list[dict]): List of segments from whisper transcription, each with 'start', 'end', 'text, and 'words'.
        diarization_segments (list[tuple]): List of (start_time, end_time, speaker_label) from pyannote.audio diarization.

    Returns:
        list[dict]: A list of combined segments, each with 'start', 'end', 'speaker', and 'text'.
    """
    aligned_transcript = []
    speaker_idx = 0

    for w_segment in whisper_segments:
        w_start = w_segment['start']
        w_end = w_segment['end']
        w_text = w_segment['text']
        
        current_speaker = "UNKNOWN"

        best_speaker = None
        max_overlap = 0.0

        for d_start, d_end, speaker_label in diarization_segments:
            overlap_start = max(w_start, d_start)
            overlap_end = min(w_end, d_end)

            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker_label

        if best_speaker:
            current_speaker = best_speaker

        aligned_transcript.append({
            "start": w_start,
            "end": w_end,
            "speaker": current_speaker,
            "text": w_text.strip()
        })
    return aligned_transcript


if __name__ == "__main__":
    sample_audio_file = "data/sample_audio/meeting.wav"

    if not os.path.exists(sample_audio_file):
        print(f"Error: Sample audio file not found at {sample_audio_file}. Please ensure it exists.")
    else:
        print(f"Processing {sample_audio_file} for transcription and diarization...")
        try:
            # Step 1: Transcribe the audio
            print("\n--- Performing Transcription ---")
            whisper_result = transcribe_audio(sample_audio_file)
            whisper_segments = whisper_result["segments"]
            print(f"Transcription complete. Found {len(whisper_segments)} segments.")

            # Step 2: Perform Diarization
            print("\n--- Performing Diarization ---")
            diarization_segments = perform_diarization(sample_audio_file)
            print(f"Diarization complete. Found {len(diarization_segments)} speaker turns.")

            # Step 3: Align Transcription with Diarization
            print("\n--- Aligning Transcription and Diarization ---")
            speaker_tagged_transcript = align_transcription_with_diarization(
                whisper_segments, diarization_segments
            )
            print("Alignment complete. \n")

            print("\n--- Speaker-Tagged Transcript ---")
            for segment in speaker_tagged_transcript:
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}")

        except ValueError as e:
            print(f"Configuration Error: {e}")
        except Exception as e:
            print(f"An error occurred during diarization: {e}")