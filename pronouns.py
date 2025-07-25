'''import whisper
import re
from Levenshtein import distance

model = whisper.load_model("medium")

audio_path = "../data/audio/wav/Aaa.wav"
expected_text = "aa"

def normalize(text):
    return re.sub(r'[^a-z]', '', text.strip().lower())

result = model.transcribe(audio_path)
predicted_text = normalize(result["text"])

expected_text = normalize(expected_text)
print(f"\n🔊 Transcribed Text: {predicted_text}")
print(f"🎯 Expected Text:    {expected_text}")

tolerance_threshold = 1

if distance(predicted_text, expected_text) <= tolerance_threshold:
    print("✅ Pronunciation is correct or close enough.")
else:
    print("❌ Pronunciation mismatch.")'''



import whisper
import re
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from Levenshtein import distance

SAMPLE_RATE = 16000  
DURATION = 3     

def normalize(text):
    return re.sub(r'[^a-z]', '', text.strip().lower())

def record_audio(file_path):
    print(f"\n🎙️ Recording for {DURATION} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(file_path, SAMPLE_RATE, audio)
    print(f"✅ Audio saved to {file_path}")

def main():
    expected_text = input("\nEnter the expected pronunciation word (default = 'aa'): ").strip().lower() or "aa"

    output_path = "user_input.wav"

    record_audio(output_path)

    print("🧠 Loading Whisper model...")
    model = whisper.load_model("medium")

    print("🔍 Transcribing...")
    result = model.transcribe(output_path)
    predicted_text = normalize(result["text"])
    expected_text = normalize(expected_text)

    # Output transcription
    print(f"\n🗣️  Transcribed Text: {predicted_text}")
    print(f"🎯 Expected Text:    {expected_text}")

    tolerance_threshold = 1
    if distance(predicted_text, expected_text) <= tolerance_threshold:
        print("✅ Pronunciation is correct or close enough.")
    else:
        print("❌ Pronunciation mismatch.")

    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    main()

