import spacy
import re
from typing import List, Dict, Optional


nlp = spacy.load("en_core_web_sm")

def extract_action_items(transcript_segments: List[Dict]) -> List[Dict]:
    """
    Extracts action items from a speaker-tagged transcript.
    Uses rule-based approach with spaCy for verb identification and NER.

    Args:
        transcript_segments (List[Dict]): A list of segments, each with 'start', 'end','speaker', and 'text'.

    Returns:
        List[Dict]: A list of dictionaries, each representing an action item with 'action', 'owner' (proposed), and 'due_date' (proposed).
    """
    action_items = []
    action_keywords = ["action", "need to", "should", "will", "let's", "we need to"]
    action_pattern = re.compile(r'\b(?:' + '|'.join(action_keywords) + r')\s+(\w+)\b.*?(?=\.|\?|\!|$)', re.IGNORECASE)

    for segment in transcript_segments:
        text = segment['text']
        speaker = segment['speaker']
        doc = nlp(text)

        
        for sent in doc.sents:
            sent_text = sent.text
            match = action_pattern.search(sent_text)
            if match:
                action_phrase = match.group(0).strip()
                
               
                owner: Optional[str] = None
                due_date: Optional[str] = None

                for ent in sent.ents:
                    if ent.label_ == "PERSON":
                        owner = ent.text
                    elif ent.label_ == "DATE":
                        due_date = ent.text
                
                
                if not owner:
                    owner = speaker if speaker != "UNKNOWN" else "Unassigned"

                action_items.append({
                    "action": action_phrase,
                    "owner": owner,
                    "due_date": due_date,
                    "source_segment": segment 
                })
    return action_items

def summarize_meeting(transcript_segments: List[Dict], num_bullets: int = 3) -> List[str]:
    """
    Generates a concise meeting summary from the speaker-tagged transcript.
    For MVP, this will be a simple extractive summary (e.g., first few sentences
    or sentences containing important keywords).

    Args:
        transcript_segments (List[Dict]): A list of segments, each with 'start', 'end',
                                         'speaker', and 'text'.
        num_bullets (int): The desired number of bullet points for the summary.

    Returns:
        List[str]: A list of strings, each representing a bullet point in the summary.
    """
    full_text = " ".join([s['text'] for s in transcript_segments])
    doc = nlp(full_text)

    summary_sentences = []
    for i, sent in enumerate(doc.sents):
        if i < num_bullets:
            summary_sentences.append(f"- {sent.text.strip()}")
        else:
            break
            
    if not summary_sentences and full_text.strip():
        summary_sentences.append(f"- {full_text.strip()}")

    return summary_sentences

if __name__ == "__main__":
    dummy_transcript_segments = [
        {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello team, we need to discuss the project deadlines. John, you should update the client by next Friday."},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Sure, I will take care of that. Also, let's schedule a follow-up meeting for next week."},
        {"start": 11.0, "end": 15.0, "speaker": "SPEAKER_00", "text": "Good idea. Mary, can you prepare the Q3 report by October 30th?"},
        {"start": 16.0, "end": 20.0, "speaker": "SPEAKER_02", "text": "Yes, I will finish the report by then. The action items are clear."},
        {"start": 21.0, "end": 25.0, "speaker": "SPEAKER_00", "text": "Excellent. We must finalize the budget by end of day tomorrow."}
    ]

    print("--- Extracting Action Items ---")
    actions = extract_action_items(dummy_transcript_segments)
    for action in actions:
        print(f"Action: {action['action']}, Owner: {action['owner']}, Due Date: {action['due_date']}")

    print("\n--- Generating Summary ---")
    summary = summarize_meeting(dummy_transcript_segments, num_bullets=3)
    for bullet in summary:
        print(bullet)