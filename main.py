import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from groq import Groq
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
import json
import uvicorn
from pydub import AudioSegment
import librosa
import numpy as np
from typing import Optional

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY") 
groq_client = Groq(api_key=GROQ_KEY)

# FFmpeg Paths
# ffmpeg_path = r"C:\ffmpeg\bin"
# os.environ["PATH"] += os.pathsep + ffmpeg_path
# AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
# AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

app = FastAPI()

print("Loading BirdNET Model...")
analyzer = Analyzer()
print("✅ Model loaded successfully.")

def identify_birds_offline(file_path):
    try:
        recording = Recording(analyzer, file_path)
        recording.analyze()
        
        detected_birds = []
        if recording.detections:
            for d in recording.detections:
                if d['confidence'] > 0.15:
                    detected_birds.append({
                        "common_name": d['common_name'],
                        "confidence": round(float(d['confidence'] * 100), 2)
                    })
            
            if not detected_birds:
                return [{"common_name": "House Sparrow", "confidence": 0}]

            unique_birds = list({v['common_name']: v for v in detected_birds}.values())
            unique_birds = sorted(unique_birds, key=lambda x: x['confidence'], reverse=True)
            return unique_birds
            
        return [{"common_name": "House Sparrow", "confidence": 0}]
    except Exception as e:
        print(f"❌ BirdNET Error: {e}")
        return [{"common_name": "House Sparrow", "confidence": 0}]

@app.post("/analyze_bird")
async def analyze_bird(
    file: UploadFile = File(...), 
    bird_name: Optional[str] = Form(None) 
):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    hz = 0 
    try:
        # 1. Frequency Analysis
        y, sr = librosa.load(file_path, sr=None)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        hz = round(float(np.mean(pitches[pitches > 0])), 2) if np.any(pitches > 0) else 0

        # 2. Multi-Bird Identification
        detected_list = identify_birds_offline(file_path)
        
        # Insaan ki awaz check karna
        is_human_present = any("Human" in b['common_name'] or "Speech" in b['common_name'] for b in detected_list)
        
        # ✅ Logic: Selection Priority
        if bird_name and any(bird_name.lower() in b['common_name'].lower() for b in detected_list):
            primary_bird = bird_name
        else:
            primary_bird = detected_list[0]['common_name']
            
        print(f"🔍 DEBUG: Target Bird for Report -> {primary_bird}")
        
        all_names = ", ".join([b['common_name'] for b in detected_list])

        # 3. Full Detailed AI Report Generation
        prompt = f"""
Analyze the bird: {primary_bird} with frequency {hz} Hz.
Note: Other sounds/birds detected: {all_names}.
Human Voice Presence: {is_human_present}.

Provide an extensive, premium ornithological report in STRICT JSON.

STRICT JSON RULES:
1. All values MUST be strings enclosed in double quotes.
2. NO comments (//) allowed.

CONTENT ENGINE (CRITICAL):
- "bird_story": Write a captivating 100-120 word narrative. 
  Treat {primary_bird} as a legendary character. If other birds like {all_names} are present, 
  briefly weave them into this cinematic legend.
- "faqs": Generate EXACTLY 10 UNIQUE & surprising FAQs. 

JSON Structure:
{{
  "name": "{primary_bird}",
  "scientific_name": "...",
  "health_status": "...",
  "health_reason": "...",
  "conservation_status": "...",
  "wingspan": "...",
  "weight": "...",
  "lifespan": "...",
  "habitat": "...",
  "diet_plan": "...",
  "best_time": "...",
  "weather": "...",
  "migration": "...",
  "behavior_traits": "...",
  "call_meaning": "...",
  "fun_fact": "...",
  "nest_style": "...",
  "nest_location": "...",
  "egg_details": "...",
  "baby_care": "...",
  "incubation": "...",
  "bird_story": "...",
  "faqs": [
    {{"question": "Q1", "answer": "A1"}},
    {{"question": "Q10", "answer": "A10"}}
  ]
}}
"""
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a specialized ornithologist. You output ONLY pure JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        ai_data = json.loads(chat_completion.choices[0].message.content)

        return {
            "is_multiple": len(detected_list) > 1,
            "human_detected": is_human_present,
            "detected_birds": detected_list, 
            "details": {
                "name": ai_data.get("name", primary_bird),
                "scientific_name": ai_data.get("scientific_name", "N/A"),
                "health_status": ai_data.get("health_status", "Stable"),
                "health_reason": ai_data.get("health_reason", "Vocal patterns are rhythmic."),
                "conservation_status": ai_data.get("conservation_status", "Least Concern"),
                "wingspan": ai_data.get("wingspan", "N/A"),
                "weight": ai_data.get("weight", "N/A"),
                "lifespan": ai_data.get("lifespan", "N/A"),
                "habitat": ai_data.get("habitat", "N/A"),
                "diet_plan": ai_data.get("diet_plan", "Insects and seeds"),
                "best_time": ai_data.get("best_time", "Dawn/Dusk"),
                "weather": ai_data.get("weather", "Sunny/Mild"),
                "migration": ai_data.get("migration", "Resident"),
                "behavior_traits": ai_data.get("behavior_traits", "Active"),
                "call_meaning": ai_data.get("call_meaning", "Social communication"),
                "fun_fact": ai_data.get("fun_fact", "Interesting bird!"),
                "nest_style": ai_data.get("nest_style", "Cup"),
                "nest_location": ai_data.get("nest_location", "Trees"),
                "egg_details": ai_data.get("egg_details", "Small eggs"),
                "baby_care": ai_data.get("baby_care", "Parental care provided"),
                "incubation": ai_data.get("incubation", "14 days"),
                "bird_story": ai_data.get("bird_story", "Once upon a time..."),
                "faqs": ai_data.get("faqs", []),
                "pitch_hz": hz
            }
        }

    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return {"error": str(e), "detected_birds": [], "details": None}
    
    finally:
        if os.path.exists(file_path): 
            try: os.remove(file_path)
            except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
