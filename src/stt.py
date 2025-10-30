import whisper   # pyright: ignore[reportMissingImports]
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes an audio file using OpenAI's Whisper model and returns a timestamped transcript.

    Args:
        audio_path (str): The path to the audio file (e.g., .mp3, .wav).

    Returns:
        dict: A dictionary containing the transcription result, including 'segments' with 
        text and timestamp information.
    """
    model =whisper.load_model("small")
    result = model.transcribe(audio_path, verbose=False, word_timestamps=True)
    
    return result

if __name__ == "__main__":
    sample_audio_file = "data/sample_audio/meeting.mp3"
    print(f"Transcribing {sample_audio_file}...")
    transcript_result = transcribe_audio(sample_audio_file) 

    print("\n--- Full Transcript ---")
    for segment in transcript_result["segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")