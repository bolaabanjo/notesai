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

if __name__ == "__main__":
    sample_audio_file = "data/sample_audio/meeting.wav"

    if not os.path.exists(sample_audio_file):
        print(f"Error: Sample audio file not found at {sample_audio_file}. Please ensure it exists.")
    else:
        print(f"Performing diarization on {sample_audio_file}...")
        try:
            diarization_result = perform_diarization(sample_audio_file)
            print("\n--- Diarization Result ---")
            for start, end, speaker in diarization_result:
                print(f"[{start:.2f}s - {end:.2f}s] {speaker}")
        except ValueError as e:
            print(f"Configuration Error: {e}")
        except Exception as e:
            print(f"An error occurred during diarization: {e}")