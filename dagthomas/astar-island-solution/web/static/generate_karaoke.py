"""Generate word-level SRT and karaoke JSON from song.mp3 using OpenAI Whisper."""
import whisper
import json
import os

SONG_PATH = os.path.join(os.path.dirname(__file__), "song.mp3")
SRT_OUT = os.path.join(os.path.dirname(__file__), "song_words.srt")
JSON_OUT = os.path.join(os.path.dirname(__file__), "song_karaoke.json")

def fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

print("Loading Whisper large-v3 model...")
model = whisper.load_model("large-v3")

print(f"Transcribing {SONG_PATH} with word timestamps...")
result = model.transcribe(SONG_PATH, word_timestamps=True, language="en")

# Build per-word SRT
srt_lines = []
idx = 1
for segment in result["segments"]:
    for word in segment.get("words", []):
        text = word["word"].strip()
        if not text:
            continue
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{fmt_time(word['start'])} --> {fmt_time(word['end'])}")
        srt_lines.append(text)
        srt_lines.append("")
        idx += 1

with open(SRT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))
print(f"Wrote {idx - 1} word cues to {SRT_OUT}")

# Build karaoke JSON (lines with per-word timing)
karaoke = []
for seg in result["segments"]:
    words = [{"word": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3)}
             for w in seg.get("words", []) if w["word"].strip()]
    if words:
        karaoke.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": words
        })

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(karaoke, f, indent=2)
print(f"Wrote {len(karaoke)} lines to {JSON_OUT}")
