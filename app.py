from flask import Flask, render_template, request, jsonify
from pitch import EnhancedPitchAnalyzer
import os
from pydub import AudioSegment

app = Flask(__name__)
analyzer = EnhancedPitchAnalyzer("hindi_pitch_dataset.csv")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/analyze_pronunciation', methods=['POST'])
def analyze():
    if "audio" not in request.files or "target" not in request.form:
        return jsonify({"success": False, "message": "Missing audio or target"}), 400

    file = request.files["audio"]
    target = request.form["target"]

    os.makedirs("uploads", exist_ok=True)

    # Step 1: Save uploaded blob (it's actually webm, not real wav!)
    webm_path = f"uploads/{target}.webm"
    file.save(webm_path)

    # Step 2: Convert webm → wav
    try:
        audio = AudioSegment.from_file(webm_path)
        wav_path = f"uploads/{target}.wav"
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return jsonify({"success": False, "message": "Conversion failed"}), 500

    # Step 3: Analyze
    results = analyzer.analyze_pronunciation(target, audio_path=wav_path)

    if results:
        return jsonify({
            "success": True,
            "feedback": results['feedback'],
            "score": results['feedback']['composite_score'],
            "level": results['feedback']['level']
        })
    else:
        return jsonify({"success": False, "message": "Analysis failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
