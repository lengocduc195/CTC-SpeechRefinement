import whisper

# Load Whisper model
model = whisper.load_model("small")

# Transcribe the audio file
result = model.transcribe("/data/test1/test1_01.wav")

# Save to .txt file
with open("/data/test1/test1_01.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription saved to transcription_result.txt")