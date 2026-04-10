import os
import io
import uuid
import json
import asyncio
import hashlib
import tempfile
import threading
from typing import Annotated, Any

import cv2
import uvicorn
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ────────────────────── TTS Engine Setup ──────────────────────
TTS_ENGINE = None
TTS_TYPE = None  # 'kokoro' or 'pyttsx3'
TTS_LOCK = threading.Lock()

# Audio cache directory
TTS_CACHE_DIR = "uploads/tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np_audio
    TTS_ENGINE = KPipeline(lang_code='a')  # American English
    TTS_TYPE = 'kokoro'
    print("✓ Kokoro TTS engine loaded (82M params, human-like voice)")
except Exception as e:
    print(f"⚠ Kokoro TTS unavailable ({e}), trying pyttsx3...")
    try:
        import pyttsx3
        TTS_ENGINE = pyttsx3.init()
        TTS_ENGINE.setProperty('rate', 165)   # Slightly slower for clarity
        TTS_ENGINE.setProperty('volume', 1.0)
        TTS_TYPE = 'pyttsx3'
        print("✓ pyttsx3 TTS engine loaded (OS-native fallback)")
    except Exception as e2:
        print(f"✗ No TTS engine available: {e2}")

def generate_tts_audio(text: str, voice: str = 'af_heart', speed: float = 1.0) -> str | None:
    """Generate TTS audio and return the cached file path. Thread-safe."""
    if not TTS_ENGINE:
        return None

    # Create a cache key from the text + voice + speed
    cache_key = hashlib.md5(f"{text}_{voice}_{speed}_{TTS_TYPE}".encode()).hexdigest()
    cache_path = os.path.join(TTS_CACHE_DIR, f"{cache_key}.wav")

    # Return cached file if it exists
    if os.path.exists(cache_path):
        return cache_path

    with TTS_LOCK:
        # Double-check after acquiring lock
        if os.path.exists(cache_path):
            return cache_path

        try:
            if TTS_TYPE == 'kokoro':
                generator = TTS_ENGINE(text, voice=voice, speed=speed)
                all_audio = []
                for _, _, audio_chunk in generator:
                    all_audio.append(audio_chunk)
                if all_audio:
                    import numpy as np_concat
                    full_audio = np_concat.concatenate(all_audio)
                    sf.write(cache_path, full_audio, 24000)
                    return cache_path

            elif TTS_TYPE == 'pyttsx3':
                TTS_ENGINE.save_to_file(text, cache_path)
                TTS_ENGINE.runAndWait()
                if os.path.exists(cache_path):
                    return cache_path

        except Exception as e:
            print(f"[TTS] Generation error: {e}")
            return None

    return None

# ────────────────────── App Setup ──────────────────────
app = FastAPI(title="Road Safety AI - Intelligent Advisory System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────── Model Loading ──────────────────────
MODELS = {}
MODEL_PATHS = {
    "gtsrb": "models/traffic_signs/gtsrb_best.pt",
    "gtsrb_v2": "models/gtsrb_v2/best.pt",
    "gtsrb_fast": "models/gtsrb_fast/best.pt",
    "road_damage": "models/road_damage/best.pt",
    "indian_signs": "models/indian_signs/best.pt",
}

# COCO road-relevant class IDs for filtering
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck,
# 9=traffic light, 11=stop sign
COCO_ROAD_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]

try:
    if YOLO:
        # Load custom-trained models
        for key, path in MODEL_PATHS.items():
            if os.path.exists(path):
                MODELS[key] = YOLO(path)
                print(f"✓ Loaded model '{key}' from {path}")
            else:
                print(f"⚠ Model not found: {path}")

        # Load pretrained COCO model (auto-downloads yolov8n.pt ~6MB)
        MODELS['coco_road'] = YOLO("yolov8n.pt")
        print("✓ Loaded COCO Road Detection model (yolov8n.pt)")

        WEATHER_MODEL = YOLO("models/weather_ai.pt") if os.path.exists("models/weather_ai.pt") else None
        if WEATHER_MODEL:
            print("✓ Loaded AI Visual Weather Detection model (weather_ai.pt)")
except Exception as e:
    print(f"Error loading models: {e}")
    WEATHER_MODEL = None

def get_ai_weather(img) -> str | None:
    if not WEATHER_MODEL: return None
    try:
        results = WEATHER_MODEL(img, verbose=False)
        top1 = results[0].probs.top1
        return results[0].names[top1].lower()
    except:
        return None

def draw_sentinel_hud(frame, detections, weather, action):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (550, 320), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    cv2.putText(frame, "SENTINEL AI TELEMETRY", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    w_text = weather.get("description", "Unknown") if weather else "Unknown"
    cv2.putText(frame, f"ENV: {w_text}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    act_text = action if action else "Path clear. Cruising."
    color = (0, 0, 255) if action else (0, 255, 0)
    cv2.putText(frame, f"CMD: {act_text}", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.putText(frame, "LIVE DETECTIONS:", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_offset = 205
    for det in detections[:4]:
        name = det.get("class_name", "Unknown").split("(")[0].strip()
        conf = det.get("confidence", 0.0) * 100
        src = "COCO" if "coco" in det.get("source", "").lower() else "GTSRB" if "gtsrb" in det.get("source", "").lower() else "AI"
        cv2.putText(frame, f"- {name} [{src}] {conf:.0f}%", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        y_offset += 30
    return frame

# Models to run when "All Models" is selected (best of each category)
ALL_MODEL_KEYS = ["gtsrb_v2", "road_damage", "coco_road"]

# ────────────────────── Weather API ──────────────────────
WEATHER_API_KEY = "b9c2bd3fe79397578499efd7a82b2766"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    try:
        resp = requests.get(WEATHER_API_URL, params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            weather_main = data.get("weather", [{}])[0].get("main", "Clear")
            weather_desc = data.get("weather", [{}])[0].get("description", "clear sky")
            temp = data.get("main", {}).get("temp", 0)
            humidity = data.get("main", {}).get("humidity", 0)
            visibility = data.get("visibility", 10000)  # meters
            wind_speed = data.get("wind", {}).get("speed", 0)  # m/s
            city = data.get("name", "Unknown")
            return {
                "status": "ok",
                "city": city,
                "weather": weather_main,
                "description": weather_desc,
                "temp": round(temp, 1),
                "humidity": humidity,
                "visibility": visibility,
                "wind_speed": round(wind_speed, 1),
                "icon": data.get("weather", [{}])[0].get("icon", "01d"),
            }
        return {"status": "error", "message": f"API returned {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ────────────────────── Intelligent Advisory Engine ──────────────────────

def load_advisories():
    try:
        with open("data/advisories.json", "r") as f:
            return json.load(f)
    except:
        return {}

ADVISORIES = load_advisories()

ROAD_DAMAGE_ADVISORIES = {
    "pothole": "Pothole detected on the road ahead. Reduce speed and steer around if safe.",
    "longitudinal_crack": "Longitudinal crack detected on road surface. Drive carefully.",
    "transverse_crack": "Transverse crack detected. Maintain steady speed.",
    "alligator_crack": "Severe alligator cracking detected. Road surface is deteriorating. Slow down.",
    "D00": "Longitudinal crack detected on road surface. Drive carefully.",
    "D10": "Transverse crack detected. Maintain steady speed.",
    "D20": "Alligator cracking detected. Road is deteriorating. Slow down.",
    "D40": "Pothole detected on the road ahead. Reduce speed and steer around.",
    "D43": "Crosswalk damage detected. Watch for uneven surfaces.",
    "D44": "White line blurring detected. Lane markings unclear.",
}

def get_weather_risk(weather: dict) -> dict:
    """Assess road conditions risk based on weather."""
    if weather.get("status") != "ok":
        return {"level": "unknown", "factor": 1.0, "warnings": []}

    warnings = []
    risk_factor = 1.0
    condition = weather.get("weather", "Clear").lower()
    visibility = weather.get("visibility", 10000)
    wind = weather.get("wind_speed", 0)
    temp = weather.get("temp", 25)

    # Rain / Storm conditions
    if condition in ("rain", "drizzle", "shower rain"):
        risk_factor += 0.3
        warnings.append("Wet road surfaces. Increase following distance.")
    elif condition in ("thunderstorm",):
        risk_factor += 0.5
        warnings.append("Thunderstorm warning! Seek shelter if possible.")
    elif condition in ("snow",):
        risk_factor += 0.6
        warnings.append("Snow detected. Roads may be icy. Drive extremely slowly.")
    elif condition in ("mist", "fog", "haze", "smoke"):
        risk_factor += 0.4
        warnings.append("Reduced visibility due to fog/mist. Use fog lights.")

    # Visibility
    if visibility < 1000:
        risk_factor += 0.4
        warnings.append(f"Very low visibility ({visibility}m). Use headlights and slow down.")
    elif visibility < 3000:
        risk_factor += 0.2
        warnings.append(f"Moderate visibility ({visibility}m). Stay alert.")

    # Wind
    if wind > 15:
        risk_factor += 0.3
        warnings.append(f"Strong winds ({wind} m/s). Keep firm grip on steering.")
    elif wind > 10:
        risk_factor += 0.1
        warnings.append(f"Moderate winds ({wind} m/s).")

    # Temperature
    if temp < 3:
        risk_factor += 0.4
        warnings.append(f"Near-freezing temperature ({temp}°C). Watch for black ice.")
    elif temp > 42:
        risk_factor += 0.2
        warnings.append(f"Extreme heat ({temp}°C). Risk of tire blowouts.")

    level = "low"
    if risk_factor >= 1.8:
        level = "high"
    elif risk_factor >= 1.3:
        level = "moderate"

    return {"level": level, "factor": round(risk_factor, 2), "warnings": warnings}


def check_immediate_action(detections: list) -> str | None:
    blocked_regions = set()
    for det in detections:
        src = det.get("source", "")
        if src in ["coco_road", "road_damage"]:
            cxr = det.get("cx_ratio", -1)
            if cxr != -1:
                if cxr < 0.33: blocked_regions.add("left")
                elif cxr < 0.66: blocked_regions.add("center")
                else: blocked_regions.add("right")
    
    if "center" in blocked_regions:
        if "right" not in blocked_regions:
            return "Obstacle ahead in center! 👉 Move Right."
        elif "left" not in blocked_regions:
            return "Obstacle ahead in center! 👉 Move Left."
        else:
            return "Obstacles blocking path! 👉 Brake and Stop."
    elif "right" in blocked_regions:
        return "Obstacle on Right. 👉 Stay Left."
    elif "left" in blocked_regions:
        return "Obstacle on Left. �� Stay Right."
    return None

def generate_smart_advisory(detections: list, weather: dict | None, model_type: str) -> dict:
    sign_advisories = []
    damage_advisories = []
    speed_limit = None
    vehicle_ids = set()
    
    for det in detections:
        name = det.get("class_name", "").split(" (")[0].strip().lower()
        conf = det.get("confidence", 0)
        if conf < 0.25:
            continue
            
        if "speed limit" in name:
            try:
                speed_val = int(''.join(filter(str.isdigit, name)))
                if speed_limit is None or speed_val < speed_limit:
                    speed_limit = speed_val
            except: pass
            
        if model_type == "road_damage":
            adv = ROAD_DAMAGE_ADVISORIES.get(name)
            if adv and adv not in damage_advisories:
                damage_advisories.append(adv)
        else:
            adv = ADVISORIES.get(name)
            if adv and adv not in sign_advisories:
                sign_advisories.append(adv)
                
        if name in ["car", "truck", "bus", "motorcycle"]:
            # Approximate unique vehicles by tracking frames
            vehicle_ids.add(f"{name}_{det.get('frame', 0)}")
            
    weather_risk = get_weather_risk(weather) if weather else {"level": "unknown", "factor": 1.0, "warnings": []}
    
    unique_vehicles = len(vehicle_ids)
    if unique_vehicles > 40: traffic = "congested with heavy traffic"
    elif unique_vehicles > 15: traffic = "experiencing moderate traffic"
    else: traffic = "clear with light traffic"
    
    guidance = f"Road ahead is {traffic}"
    conds = []
    if damage_advisories: conds.append("poor road conditions due to damage")
    if weather_risk["level"] in ["high", "moderate"]: conds.append("adverse weather")
    
    if conds:
        guidance += f" and {' and '.join(conds)}."
    else:
        guidance += "."

    rec_speed = speed_limit if speed_limit else 60
    if weather_risk["level"] == "high" or damage_advisories:
        rec_speed = max(20, rec_speed - 20)
    elif weather_risk["level"] == "moderate" or unique_vehicles > 15:
        rec_speed = max(20, rec_speed - 10)
        
    guidance += f" Maintain a speed of {max(20, rec_speed - 10)}-{rec_speed} km/h and keep to the right for safer driving."
    
    return {
        "combined_advisory": guidance,
        "sign_alerts": sign_advisories[:5],
        "damage_alerts": damage_advisories[:5],
        "weather_risk": weather_risk,
        "speed_limit_detected": speed_limit,
    }


# ────────────────────── Static Files ──────────────────────
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ────────────────────── Data Models ──────────────────────
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    class_id: int
    representative_url: str | None = None
    full_frame_url: str | None = None

class PredictionResponse(BaseModel):
    type: str
    detections: list[DetectionResult]
    media_url: str
    advisory: dict | None = None
    weather: dict | None = None

# ────────────────────── Routes ──────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/advisories")
async def get_advisories() -> Any:
    return ADVISORIES


@app.get("/weather")
async def get_weather(lat: float = Query(13.0827), lon: float = Query(80.2707)) -> Any:
    """Get current weather. Default coords = Chennai, India."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fetch_weather, lat, lon)
    return result


# ────────────────────── Image Processing ──────────────────────

# Color palette for different models (BGR)
MODEL_COLORS = {
    "gtsrb": (0, 255, 0), "gtsrb_v2": (0, 255, 0), "gtsrb_fast": (0, 255, 0),
    "road_damage": (0, 128, 255), "indian_signs": (255, 0, 255),
    "coco_road": (255, 255, 0),
}

def run_model_on_image(model, img, model_key, annotated, conf_threshold=0.5):
    """Run a single model on image, draw boxes, return detections."""
    classes_filter = COCO_ROAD_CLASSES if model_key == 'coco_road' else None
    results = model(img, classes=classes_filter, verbose=False)
    result = results[0]
    names = model.names
    color = MODEL_COLORS.get(model_key, (0, 255, 0))
    detections = []

    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(result.boxes.conf[i])
        cls_id = int(result.boxes.cls[i])
        label = names[cls_id]
        if conf < (0.45 if model_key == 'coco_road' else 0.25):
            continue

        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "class_name": label, "confidence": round(conf, 3),
            "class_id": cls_id, "source": model_key, "cx_ratio": round(((x1 + x2) / 2.0) / annotated.shape[1], 3),
        })
    return detections


def process_image(contents: bytes, filename: str, model_type: str, weather: dict | None = None, roi: dict | None = None) -> dict:
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    ai_weather_cond = get_ai_weather(img)
    # Process only top 90% to ignore bonnet
    img = img.crop((0, 0, img.width, int(img.height * 0.9)))
    if ai_weather_cond and weather:
        weather["weather"] = ai_weather_cond
        weather["description"] = ai_weather_cond.capitalize() + " (AI Detected)"
        
    img_cv = np.array(img)[..., ::-1].copy()  # RGB to BGR
    annotated = img_cv.copy()
    full_h, full_w = img_cv.shape[:2]

    # Apply ROI cropping if provided
    roi_offset_x, roi_offset_y = 0, 0
    detection_img = img  # PIL image for model input
    detection_cv = img_cv  # CV2 image for drawing
    if roi:
        rx = int(roi['x'] * full_w)
        ry = int(roi['y'] * full_h)
        rw = int(roi['w'] * full_w)
        rh = int(roi['h'] * full_h)
        # Clamp
        rx = max(0, min(rx, full_w - 1))
        ry = max(0, min(ry, full_h - 1))
        rw = min(rw, full_w - rx)
        rh = min(rh, full_h - ry)
        if rw > 10 and rh > 10:
            roi_offset_x, roi_offset_y = rx, ry
            detection_img = img.crop((rx, ry, rx + rw, ry + rh))
            detection_cv = img_cv[ry:ry+rh, rx:rx+rw].copy()
            # Draw ROI boundary on annotated full image
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (59, 130, 246), 2)
            cv2.putText(annotated, "ROI", (rx + 4, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 1, cv2.LINE_AA)
            # Dim outside ROI
            overlay = annotated.copy()
            mask = np.ones_like(overlay, dtype=np.uint8) * 40
            mask[ry:ry+rh, rx:rx+rw] = 0
            annotated = cv2.subtract(annotated, mask)

    all_detections = []

    if model_type == 'all_models':
        # Run ALL models on ROI region
        for mkey in ALL_MODEL_KEYS:
            m = MODELS.get(mkey)
            if m:
                # We run on the cropped ROI image, then offset bounding boxes back
                roi_annotated = detection_cv.copy()
                dets = run_model_on_image(m, detection_img, mkey, roi_annotated)
                # Offset detections and paste annotations back onto full annotated image
                for det in dets:
                    det['cx_ratio'] = round(((det.get('cx_ratio', 0.5) * detection_cv.shape[1]) + roi_offset_x) / full_w, 3)
                all_detections.extend(dets)
                # Paste ROI annotations back
                if roi and roi_offset_x > 0 or roi_offset_y > 0:
                    annotated[roi_offset_y:roi_offset_y+detection_cv.shape[0], roi_offset_x:roi_offset_x+detection_cv.shape[1]] = roi_annotated
                else:
                    annotated = roi_annotated
    else:
        # Run selected model on ROI
        model = MODELS.get(model_type)
        if model:
            results = model(detection_img)
            result = results[0]
            roi_annotated = result.plot()
            names = result.names
            for i in range(len(result.boxes)):
                conf = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                if conf < 0.25: continue
                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                bx1, by1, bx2, by2 = box
                cx = ((bx1 + bx2) / 2.0 + roi_offset_x) / full_w
                all_detections.append({
                    "class_name": names[cls_id], "confidence": round(conf, 3),
                    "class_id": cls_id, "source": model_type, "cx_ratio": round(cx, 3),
                })
            # Paste ROI annotations back
            if roi and (roi_offset_x > 0 or roi_offset_y > 0):
                annotated[roi_offset_y:roi_offset_y+roi_annotated.shape[0], roi_offset_x:roi_offset_x+roi_annotated.shape[1]] = roi_annotated
            else:
                annotated = roi_annotated

        # Also run COCO in background
        coco_model = MODELS.get('coco_road')
        if coco_model and model_type != 'coco_road':
            roi_annotated_coco = annotated[roi_offset_y:roi_offset_y+detection_cv.shape[0], roi_offset_x:roi_offset_x+detection_cv.shape[1]].copy() if roi else annotated.copy()
            coco_dets = run_model_on_image(coco_model, detection_img, 'coco_road', roi_annotated_coco)
            for det in coco_dets:
                det['cx_ratio'] = round(((det.get('cx_ratio', 0.5) * detection_cv.shape[1]) + roi_offset_x) / full_w, 3)
            all_detections.extend(coco_dets)
            if roi and (roi_offset_x > 0 or roi_offset_y > 0):
                annotated[roi_offset_y:roi_offset_y+detection_cv.shape[0], roi_offset_x:roi_offset_x+detection_cv.shape[1]] = roi_annotated_coco
            else:
                annotated = roi_annotated_coco

    # Save annotated image
    annotated_img = Image.fromarray(annotated[..., ::-1])
    # Bake HUD into static image
    action_text = check_immediate_action(all_detections)
    hud_frame = draw_sentinel_hud(np.array(annotated_img)[..., ::-1], all_detections, weather, action_text)
    annotated_img = Image.fromarray(hud_frame[..., ::-1])
    
    safe_filename = filename.replace(" ", "_")
    temp_path = f"uploads/{safe_filename}"
    annotated_img.save(temp_path)

    advisory = generate_smart_advisory(all_detections, weather, model_type)

    return {
        "type": "image",
        "detections": all_detections,
        "media_url": f"/results/{safe_filename}",
        "advisory": advisory,
        "weather": weather,
    }


# ────────────────────── Video Processing ──────────────────────

def process_video_generator(input_path: str, output_path: str, safe_filename: str, unique_id: str, model_type: str, weather: dict | None = None) -> dict:
    is_all = model_type == 'all_models'
    models_to_run = [(k, MODELS[k]) for k in ALL_MODEL_KEYS if k in MODELS] if is_all else []
    model = MODELS.get(model_type) if not is_all else None
    coco_model = MODELS.get('coco_road') if not is_all else None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, int(height * 0.9)))

    frame_count = 0
    max_frames = 300
    all_detections_history = []

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break
            
        frame = frame[:int(height * 0.9), :] # Crop top 60%

        if frame_count == 0:
            pil_frame_first = Image.fromarray(frame[..., ::-1])
            ai_weather_cond = get_ai_weather(pil_frame_first)
            if ai_weather_cond and weather:
                weather["weather"] = ai_weather_cond
                weather["description"] = ai_weather_cond.capitalize() + " (AI Detected)"

        if is_all:
            annotated_frame = frame.copy()
            pil_frame = Image.fromarray(frame[..., ::-1])
            for mkey, m in models_to_run:
                classes_filter = COCO_ROAD_CLASSES if mkey == 'coco_road' else None
                results = m(pil_frame, classes=classes_filter, verbose=False)
                result = results[0]
                mnames = m.names
                color = MODEL_COLORS.get(mkey, (0, 255, 0))

                for j in range(len(result.boxes)):
                    box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                    conf = float(result.boxes.conf[j])
                    cls_id = int(result.boxes.cls[j])
                    label = mnames[cls_id]
                    if conf < (0.45 if mkey == 'coco_road' else 0.25):
                        continue
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 2, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    # Save crop and full frame for telemetry log
                    cy1c, cy2c = max(0, y1), min(height, y2)
                    cx1c, cx2c = max(0, x1), min(width, x2)
                    rep_url = None
                    full_url = None
                    if cy2c > cy1c and cx2c > cx1c:
                        crop_img = frame[cy1c:cy2c, cx1c:cx2c]
                        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                            rep_fn = f"crop_{unique_id}_f{frame_count}_{mkey}_{j}.jpg"
                            full_fn = f"full_{unique_id}_f{frame_count}_{mkey}_{j}.jpg"
                            cv2.imwrite(f"uploads/{rep_fn}", crop_img)
                            cv2.imwrite(f"uploads/{full_fn}", annotated_frame)
                            rep_url = f"/results/{rep_fn}"
                            full_url = f"/results/{full_fn}"

                    all_detections_history.append({
                        "class_name": f"{label} (Frame {frame_count})",
                        "confidence": round(conf, 3), "class_id": cls_id, "source": mkey, "cx_ratio": round(((x1 + x2) / 2.0) / width, 3),
                        "representative_url": rep_url, "full_frame_url": full_url,
                    })
        else:
            results = model(frame, verbose=False)
            result = results[0]
            annotated_frame = result.plot()
            names = result.names

            if coco_model and model_type != 'coco_road':
                coco_results = coco_model(frame, verbose=False, classes=COCO_ROAD_CLASSES)
                coco_result = coco_results[0]
                coco_names = coco_model.names
                for j in range(len(coco_result.boxes)):
                    cbox = coco_result.boxes.xyxy[j].cpu().numpy().astype(int)
                    cconf = float(coco_result.boxes.conf[j])
                    ccls = int(coco_result.boxes.cls[j])
                    clabel = coco_names[ccls]
                    if cconf < 0.45:
                        continue
                    cx1, cy1, cx2, cy2 = cbox
                    cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                    ctext = f"{clabel} {cconf:.2f}"
                    (tw, th), _ = cv2.getTextSize(ctext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (cx1, cy1 - th - 6), (cx1 + tw + 4, cy1), (255, 255, 0), -1)
                    cv2.putText(annotated_frame, ctext, (cx1 + 2, cy1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    if frame_count % 10 == 0:
                        all_detections_history.append({
                            "class_name": f"{clabel} (Frame {frame_count})",
                            "confidence": cconf, "class_id": ccls, "source": "coco_road", "cx_ratio": round(((cbox[0] + cbox[2]) / 2.0) / width, 3) if "cbox" in locals() else round(((cx1 + cx2) / 2.0) / width, 3),
                        })

            class_ids = result.boxes.cls.cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
            for i, cls_id in enumerate(class_ids):
                c_name = names[int(cls_id)]
                conf = confidences[i]
                box = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                y1, y2 = max(0, y1), min(height, y2)
                x1, x2 = max(0, x1), min(width, x2)
                if y2 > y1 and x2 > x1:
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                        rep_filename = f"crop_{unique_id}_f{frame_count}_{i}.jpg"
                        full_filename = f"full_{unique_id}_f{frame_count}_{i}.jpg"
                        cv2.imwrite(f"uploads/{rep_filename}", crop_img)
                        cv2.imwrite(f"uploads/{full_filename}", annotated_frame)
                        all_detections_history.append({
                            "class_name": f"{c_name} (Frame {frame_count})",
                            "confidence": float(conf), "class_id": int(cls_id),
                            "source": model_type, "cx_ratio": round(((x1 + x2) / 2.0) / img.width, 3) if "img" in locals() else round(((x1 + x2) / 2.0) / width, 3),
                            "representative_url": f"/results/{rep_filename}",
                            "full_frame_url": f"/results/{full_filename}"
                        })

        current_frame_dets = [d for d in all_detections_history if f"Frame {frame_count}" in d.get("class_name", "")]
        annotated_frame = draw_sentinel_hud(annotated_frame, current_frame_dets, weather, check_immediate_action(current_frame_dets))
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    advisory = generate_smart_advisory(all_detections_history, weather, model_type)

    output_filename = os.path.basename(output_path)
    return {
        "type": "video",
        "detections": all_detections_history,
        "media_url": f"/results/{output_filename}",
        "advisory": advisory,
        "weather": weather,
    }


# ────────────────────── Predict Endpoint ──────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    model_type: Annotated[str, Form()] = "gtsrb",
    lat: Annotated[float, Form()] = 13.0827,
    lon: Annotated[float, Form()] = 80.2707,
    roi_x: Annotated[float | None, Form()] = None,
    roi_y: Annotated[float | None, Form()] = None,
    roi_w: Annotated[float | None, Form()] = None,
    roi_h: Annotated[float | None, Form()] = None,
) -> Any:

    if model_type not in MODELS and model_type != 'all_models':
        return JSONResponse({"error": f"Model '{model_type}' is not loaded."}, status_code=500)

    # Parse ROI
    roi = None
    if roi_x is not None and roi_y is not None and roi_w is not None and roi_h is not None:
        roi = {'x': roi_x, 'y': roi_y, 'w': roi_w, 'h': roi_h}
        print(f"[ROI] Detection region: x={roi_x:.3f}, y={roi_y:.3f}, w={roi_w:.3f}, h={roi_h:.3f}")

    # Fetch weather in parallel
    loop = asyncio.get_event_loop()
    weather = await loop.run_in_executor(None, fetch_weather, lat, lon)

    content_type = file.content_type

    if content_type.startswith("image/"):
        contents = await file.read()
        try:
            result = await loop.run_in_executor(None, process_image, contents, file.filename, model_type, weather, roi)
            return result
        except Exception as e:
            return JSONResponse({"error": f"Image processing failed: {e}"}, status_code=400)

    elif content_type.startswith("video/"):
        safe_filename = file.filename.replace(" ", "_")
        unique_id = str(uuid.uuid4())[:8]
        input_path = f"uploads/in_{unique_id}_{safe_filename}"

        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        output_filename = f"out_{unique_id}_{safe_filename}"
        if not output_filename.endswith(".mp4"):
            output_filename = output_filename.rsplit(".", 1)[0] + ".mp4"

        output_path = f"uploads/{output_filename}"

        try:
            result = await loop.run_in_executor(
                None,
                process_video_generator,
                input_path, output_path, safe_filename, unique_id, model_type, weather
            )
            return result
        except Exception as e:
            return JSONResponse({"error": f"Video processing failed: {e}"}, status_code=500)

    else:
        return JSONResponse({"error": "Unsupported file type."}, status_code=400)


# ────────────────────── Streaming Video Endpoint ──────────────────────

@app.post("/predict-stream")
async def predict_stream(
    file: Annotated[UploadFile, File(...)],
    model_type: Annotated[str, Form()] = "gtsrb",
    lat: Annotated[float, Form()] = 13.0827,
    lon: Annotated[float, Form()] = 80.2707,
    roi_x: Annotated[float | None, Form()] = None,
    roi_y: Annotated[float | None, Form()] = None,
    roi_w: Annotated[float | None, Form()] = None,
    roi_h: Annotated[float | None, Form()] = None,
):
    """Stream detection events in real-time as video frames are processed."""

    if model_type not in MODELS and model_type != 'all_models':
        return JSONResponse({"error": f"Model '{model_type}' is not loaded."}, status_code=500)

    # Parse ROI
    roi = None
    if roi_x is not None and roi_y is not None and roi_w is not None and roi_h is not None:
        roi = {'x': roi_x, 'y': roi_y, 'w': roi_w, 'h': roi_h}
        print(f"[ROI-Stream] Detection region: x={roi_x:.3f}, y={roi_y:.3f}, w={roi_w:.3f}, h={roi_h:.3f}")

    content_type = file.content_type

    # For images, use regular endpoint logic
    if content_type.startswith("image/"):
        contents = await file.read()
        loop = asyncio.get_event_loop()
        weather = await loop.run_in_executor(None, fetch_weather, lat, lon)
        result = await loop.run_in_executor(None, process_image, contents, file.filename, model_type, weather, roi)
        return JSONResponse(result)

    # For videos, stream detections frame-by-frame
    safe_filename = file.filename.replace(" ", "_")
    unique_id = str(uuid.uuid4())[:8]
    input_path = f"uploads/in_{unique_id}_{safe_filename}"

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    output_filename = f"out_{unique_id}_{safe_filename}"
    if not output_filename.endswith(".mp4"):
        output_filename = output_filename.rsplit(".", 1)[0] + ".mp4"
    output_path = f"uploads/{output_filename}"

    loop = asyncio.get_event_loop()
    weather = await loop.run_in_executor(None, fetch_weather, lat, lon)

    def stream_video_processing():
        is_all = model_type == 'all_models'
        models_to_run = [(k, MODELS[k]) for k in ALL_MODEL_KEYS if k in MODELS] if is_all else []
        single_model = MODELS.get(model_type) if not is_all else None
        coco_model = MODELS.get('coco_road') if not is_all else None

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            yield json.dumps({"event": "error", "message": "Failed to open video"}) + "\n"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cropped_h = int(height * 0.9)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, cropped_h))

        # Pre-compute ROI pixel coordinates for video frames
        roi_px = None
        if roi:
            rx = int(roi['x'] * width)
            ry = int(roi['y'] * cropped_h)
            rw = int(roi['w'] * width)
            rh = int(roi['h'] * cropped_h)
            rx = max(0, min(rx, width - 1))
            ry = max(0, min(ry, cropped_h - 1))
            rw = min(rw, width - rx)
            rh = min(rh, cropped_h - ry)
            if rw > 10 and rh > 10:
                roi_px = {'x': rx, 'y': ry, 'w': rw, 'h': rh}
                print(f"[ROI-Stream] Pixel ROI: x={rx}, y={ry}, w={rw}, h={rh}")

        frame_count = 0
        max_frames = 300
        all_detections = []
        announced_classes = set()

        success, frame_check = cap.read()
        if success:
            ai_cond = get_ai_weather(Image.fromarray(frame_check[..., ::-1]))
            if ai_cond and weather: 
                weather["weather"] = ai_cond
                weather["description"] = ai_cond.capitalize() + " (AI Detected)"
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        yield json.dumps({
            "event": "start",
            "total_frames": min(total_frames, max_frames),
            "weather": weather,
            "roi": roi_px is not None,
        }) + "\n"

        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break
                
            frame = frame[:cropped_h, :]  # Crop top 90%

            frame_new_detections = []
            frame_all_classes = []
            annotated_frame = frame.copy()

            # Determine detection input region — either ROI crop or full frame
            if roi_px:
                rx, ry, rw, rh = roi_px['x'], roi_px['y'], roi_px['w'], roi_px['h']
                detection_frame = frame[ry:ry+rh, rx:rx+rw]
                detection_pil = Image.fromarray(detection_frame[..., ::-1])
                # Draw ROI boundary on annotated frame
                cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), (59, 130, 246), 2)
                # Dim outside ROI
                dim_mask = np.ones_like(annotated_frame, dtype=np.uint8) * 35
                dim_mask[ry:ry+rh, rx:rx+rw] = 0
                annotated_frame = cv2.subtract(annotated_frame, dim_mask)
            else:
                detection_frame = frame
                detection_pil = Image.fromarray(frame[..., ::-1])
                rx, ry = 0, 0

            if is_all:
                # Run ALL models on detection region
                for mkey, m in models_to_run:
                    classes_filter = COCO_ROAD_CLASSES if mkey == 'coco_road' else None
                    results = m(detection_pil, classes=classes_filter, verbose=False)
                    result = results[0]
                    mnames = m.names
                    color = MODEL_COLORS.get(mkey, (0, 255, 0))

                    for j in range(len(result.boxes)):
                        box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                        conf = float(result.boxes.conf[j])
                        cls_id = int(result.boxes.cls[j])
                        label = mnames[cls_id]
                        if conf < (0.45 if mkey == 'coco_road' else 0.25):
                            continue

                        frame_all_classes.append(label)

                        # Offset box coordinates back to full frame
                        bx1, by1, bx2, by2 = box
                        x1, y1, x2, y2 = bx1 + rx, by1 + ry, bx2 + rx, by2 + ry
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(annotated_frame, text, (x1 + 2, y1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        if label not in announced_classes:
                            announced_classes.add(label)
                            # Save crop for telemetry thumbnail
                            cy1c, cy2c = max(0, y1), min(cropped_h, y2)
                            cx1c, cx2c = max(0, x1), min(width, x2)
                            rep_url = None
                            full_url = None
                            if cy2c > cy1c and cx2c > cx1c:
                                crop_img = frame[cy1c:cy2c, cx1c:cx2c]
                                if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                                    rep_fn = f"crop_{unique_id}_f{frame_count}_{mkey}_{j}.jpg"
                                    full_fn = f"full_{unique_id}_f{frame_count}_{mkey}_{j}.jpg"
                                    cv2.imwrite(f"uploads/{rep_fn}", crop_img)
                                    cv2.imwrite(f"uploads/{full_fn}", annotated_frame)
                                    rep_url = f"/results/{rep_fn}"
                                    full_url = f"/results/{full_fn}"
                            det = {
                                "class_name": label, "confidence": round(conf, 3),
                                "class_id": cls_id, "source": mkey, "cx_ratio": round(((x1 + x2) / 2.0) / width, 3), "frame": frame_count,
                                "representative_url": rep_url, "full_frame_url": full_url,
                            }
                            frame_new_detections.append(det)
                            all_detections.append(det)
            else:
                # Run single selected model on detection region
                results = single_model(detection_frame, verbose=False)
                result = results[0]
                roi_annotated = result.plot()
                names = result.names

                # Paste ROI annotations back onto full annotated frame
                if roi_px:
                    annotated_frame[ry:ry+rh, rx:rx+rw] = roi_annotated
                else:
                    annotated_frame = roi_annotated

                # COCO background detection on detection region
                if coco_model and model_type != 'coco_road':
                    coco_results = coco_model(detection_frame, verbose=False, classes=COCO_ROAD_CLASSES)
                    coco_result = coco_results[0]
                    coco_names = coco_model.names

                    for j in range(len(coco_result.boxes)):
                        cbox = coco_result.boxes.xyxy[j].cpu().numpy().astype(int)
                        cconf = float(coco_result.boxes.conf[j])
                        ccls = int(coco_result.boxes.cls[j])
                        clabel = coco_names[ccls]
                        if cconf < 0.45:
                            continue

                        frame_all_classes.append(clabel)

                        # Offset back to full frame
                        cbx1, cby1, cbx2, cby2 = cbox
                        cx1, cy1, cx2, cy2 = cbx1 + rx, cby1 + ry, cbx2 + rx, cby2 + ry
                        cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                        ctext = f"{clabel} {cconf:.2f}"
                        (tw, th), _ = cv2.getTextSize(ctext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (cx1, cy1 - th - 6), (cx1 + tw + 4, cy1), (255, 255, 0), -1)
                        cv2.putText(annotated_frame, ctext, (cx1 + 2, cy1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        if clabel not in announced_classes:
                            announced_classes.add(clabel)
                            det = {"class_name": clabel, "confidence": round(cconf, 3), "class_id": ccls, "source": "coco_road", "cx_ratio": round(((cx1 + cx2) / 2.0) / width, 3), "frame": frame_count}
                            frame_new_detections.append(det)
                            all_detections.append(det)

                # Selected model detections
                class_ids = result.boxes.cls.cpu().numpy().tolist()
                confidences = result.boxes.conf.cpu().numpy().tolist()

                for i, cls_id_val in enumerate(class_ids):
                    c_name = names[int(cls_id_val)]
                    conf = confidences[i]
                    if conf < (0.45 if model_type == 'coco_road' else 0.25):
                        continue
                    box = result.boxes.xyxy[i].cpu().numpy()
                    bx1, by1, bx2, by2 = map(int, box)
                    # Offset back to full frame
                    x1, y1, x2, y2 = bx1 + rx, by1 + ry, bx2 + rx, by2 + ry
                    y1, y2 = max(0, y1), min(cropped_h, y2)
                    x1, x2 = max(0, x1), min(width, x2)

                    frame_all_classes.append(c_name)

                    rep_url = None
                    full_url = None
                    if y2 > y1 and x2 > x1:
                        crop_img = frame[y1:y2, x1:x2]
                        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                            rep_fn = f"crop_{unique_id}_f{frame_count}_{i}.jpg"
                            full_fn = f"full_{unique_id}_f{frame_count}_{i}.jpg"
                            cv2.imwrite(f"uploads/{rep_fn}", crop_img)
                            cv2.imwrite(f"uploads/{full_fn}", annotated_frame)
                            rep_url = f"/results/{rep_fn}"
                            full_url = f"/results/{full_fn}"

                    if c_name not in announced_classes:
                        announced_classes.add(c_name)
                        det = {
                            "class_name": c_name, "confidence": round(float(conf), 3),
                            "class_id": int(cls_id_val), "source": model_type, "cx_ratio": round(((x1 + x2) / 2.0) / width, 3), "frame": frame_count,
                            "representative_url": rep_url, "full_frame_url": full_url,
                        }
                        frame_new_detections.append(det)
                        all_detections.append(det)

            if frame_new_detections:
                yield json.dumps({"event": "detection", "frame": frame_count, "detections": frame_new_detections}) + "\n"
                immed_action = check_immediate_action(frame_new_detections)
                if immed_action:
                    yield json.dumps({"event": "immediate_action", "action": immed_action, "frame": frame_count}) + "\n"
            else:
                immed_action = None
            
            annotated_frame = draw_sentinel_hud(annotated_frame, frame_new_detections, weather, immed_action)
            out_writer.write(annotated_frame)

            if frame_count % 5 == 0:
                yield json.dumps({"event": "speed_sync", "frame": frame_count, "classes": frame_all_classes}) + "\n"

            if frame_count % 20 == 0:
                yield json.dumps({"event": "progress", "frame": frame_count, "total": min(total_frames, max_frames)}) + "\n"

            window_frames = int(fps * 3) if fps > 0 else 90
            if frame_count > 0 and frame_count % window_frames == 0:
                recent_dets = all_detections[-300:]
                pred_adv = generate_smart_advisory(recent_dets, weather, model_type)
                yield json.dumps({"event": "predictive_advisory", "advisory": pred_adv, "frame": frame_count}) + "\n"

            frame_count += 1

        cap.release()
        out_writer.release()

        advisory = generate_smart_advisory(all_detections, weather, model_type)

        yield json.dumps({
            "event": "complete", "type": "video",
            "detections": all_detections,
            "media_url": f"/results/{os.path.basename(output_path)}",
            "advisory": advisory, "weather": weather,
            "total_frames_processed": frame_count,
        }) + "\n"

    return StreamingResponse(stream_video_processing(), media_type="application/x-ndjson")


@app.get("/results/{filename}")
async def get_result(filename: str) -> Any:
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)


# ────────────────────── TTS Endpoint ──────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"   # Default: female American voice
    speed: float = 1.0

@app.post("/tts")
async def text_to_speech(req: TTSRequest) -> Any:
    """Generate AI voice audio from text. Returns WAV audio file."""
    if not TTS_ENGINE:
        return JSONResponse({"error": "No TTS engine available"}, status_code=503)

    if not req.text or len(req.text.strip()) == 0:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    # Limit text length for safety
    text = req.text[:500]

    loop = asyncio.get_event_loop()
    audio_path = await loop.run_in_executor(None, generate_tts_audio, text, req.voice, req.speed)

    if audio_path and os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav", filename="advisory.wav")

    return JSONResponse({"error": "TTS generation failed"}, status_code=500)

@app.get("/tts/status")
async def tts_status() -> Any:
    """Check TTS engine availability and type."""
    return {
        "available": TTS_ENGINE is not None,
        "engine": TTS_TYPE,
        "voices": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                   "am_adam", "am_michael"] if TTS_TYPE == 'kokoro' else ["default"],
    }


# ────────────────────── Conversational AI Chat ──────────────────────

class ChatRequest(BaseModel):
    message: str
    context: dict = {}  # {detections, weather, advisory, speed, model}

HAS_GREETED = False
FAILED_ATTEMPTS = 0
SPEED_INSIST_COUNT = 0
LAST_RECOMMENDATION_REASON = "I haven't made a recent recommendation."

def build_chat_response(message: str, ctx: dict) -> str:
    """Generate an intelligent response based on strict priority system."""
    global LAST_RECOMMENDATION_REASON, HAS_GREETED, FAILED_ATTEMPTS, SPEED_INSIST_COUNT
    
    msg = message.lower().strip()
    words = msg.split()
    
    # Check for short unclear messages
    if len(msg) < 3 and msg not in ["hi"]:
        FAILED_ATTEMPTS += 1
        if FAILED_ATTEMPTS > 1:
            return "Try asking something like: current speed or road status."
        return "Do you mean objects or vehicles?"

    # ── Context Parsing  ──
    detections = ctx.get("detections", [])
    weather = ctx.get("weather", {})
    speed = ctx.get("speed", 60)
    
    vehicles = [d for d in detections if d.get("class_name", "").lower() in ["car", "truck", "bus", "motorcycle", "bicycle"]]
    persons = [d for d in detections if "person" in d.get("class_name", "").lower() or "pedestrian" in d.get("class_name", "").lower()]
    damage = [d for d in detections if d.get("source") == "road_damage"]
    total_objects = len(detections)

    has_weather = weather.get("status") == "ok"
    weather_desc = weather.get("description", "clear") if has_weather else "clear"
    visibility = weather.get("visibility", 10000) if has_weather else 10000

    # ── Evaluate True High Risk ──
    acute_pedestrian = any(0.35 < p.get("cx_ratio", 0.5) < 0.65 for p in persons)
    high_risk_active = False
    
    if acute_pedestrian:
        high_risk_active = True
        high_risk_override = "⚠️ Obstacle very close. Slow down immediately."
    elif visibility < 500:
        high_risk_active = True
        high_risk_override = "⚠️ Visibility extremely low. Slow down immediately."

    is_speed_query = any(w in msg for w in ["speed", "how fast", "km/h", "slow down"])
    is_increase_speed = "increase speed" in msg or "faster" in msg
    is_why_query = msg == "why" or msg == "why?"

    # ── DECISION ENGINE (Priority System) ──
    
    if is_why_query:
        return LAST_RECOMMENDATION_REASON

    # Speed Increase Logic
    if is_increase_speed:
        if total_objects > 0 or damage:
            SPEED_INSIST_COUNT += 1
            if SPEED_INSIST_COUNT > 1:
                return f"{speed} km/h is safest right now. Increasing speed may increase risk."
            LAST_RECOMMENDATION_REASON = "Because there are obstacles or damage detected ahead."
            return "You can increase speed when the road is clear and no obstacles are ahead."
        else:
            SPEED_INSIST_COUNT = 0
            LAST_RECOMMENDATION_REASON = "Because the road is currently clear."
            return "The road appears clear. You may carefully increase speed."

    # Standard Speed Logic
    if is_speed_query:
        factors = []
        if damage: factors.append("road damage")
        if vehicles: factors.append("traffic")
        if not factors: factors.append("current conditions")
        reason = " and ".join(factors)
        
        LAST_RECOMMENDATION_REASON = f"Because the road is affected by {reason}."
        base_resp = f"Recommended speed is {speed} km/h due to {reason}."
        
        if high_risk_active and acute_pedestrian:
            return f"{base_resp} Obstacle ahead, proceed carefully."
        return base_resp

    if high_risk_active:
        LAST_RECOMMENDATION_REASON = "Because there is a critical hazard detected."
        return high_risk_override

    # Handling greetings
    if any(w in msg for w in ["hello", "hi", "hey", "sentinel"]):
        if HAS_GREETED:
            return "I am ready. Ask me about the current analysis, road conditions, or speed."
        HAS_GREETED = True
        return "Hello. I am Sentinel AI, your driving co-pilot. What would you like to know about the road analysis?"

    # Full Analysis / Status Report
    if any(w in msg for w in ["status", "report", "analysis", "everything"]):
        LAST_RECOMMENDATION_REASON = "Because you requested a full analysis report."
        return f"Analysis summary: Speed is {speed} km/h. Road is {'damaged' if damage else 'good'}. Weather is {weather_desc}. {total_objects} total objects detected. Do you have any specific questions?"

    # Handling 'how many' and incomplete intents
    if "how many" in msg:
        if "car" in msg or "vehicle" in msg:
            LAST_RECOMMENDATION_REASON = f"Because I detect {len(vehicles)} vehicles."
            return f"I detect {len(vehicles)} vehicles."
        elif len(words) <= 3: # Just "how many" or "how many objects"
            if total_objects > 0:
                LAST_RECOMMENDATION_REASON = f"Because I detect {total_objects} objects."
                return f"I detect {total_objects} objects, including {len(vehicles)} vehicles."
            else:
                return "Do you mean objects or vehicles? Currently, I don't see any in front of us."
    
    if "face" in msg:
        LAST_RECOMMENDATION_REASON = "Because my primary sensors are optimized for road hazards."
        return "I don't detect faces. I detect vehicles and road objects."

    if any(w in msg for w in ["road", "condition", "surface", "damage", "pothole"]):
        cond = "damaged" if damage else "good"
        LAST_RECOMMENDATION_REASON = f"Because I observe {len(damage)} instances of damage."
        return f"Based on the analysis, the road condition is {cond}."

    if any(w in msg for w in ["weather", "rain", "fog", "visibility", "outside", "clear"]):
        LAST_RECOMMENDATION_REASON = f"Because my sensors indicate {weather_desc}."
        if visibility < 3000:
            return f"It's {weather_desc}. Visibility is low. Drive carefully."
        return f"The current weather is {weather_desc}."

    if any(w in msg for w in ["safe", "safety", "danger"]):
        if high_risk_active:
            return "It is currently unsafe due to an immediate obstacle."
        elif damage or total_objects > 3:
            return "It is moderately safe, but remain alert due to traffic or road quality."
        else:
            return "Conditions appear perfectly safe for driving."

    if any(w in msg for w in ["vehicle", "car", "see", "traffic", "object"]):
        if not total_objects:
            LAST_RECOMMENDATION_REASON = "Because there are no objects in my field of view."
            return "The road ahead appears clear."
        LAST_RECOMMENDATION_REASON = f"Because I am tracking {total_objects} objects."
        return f"Based on my analysis, I detect {total_objects} total objects, including {len(vehicles)} vehicles."

    # Dynamic lookup for specific objects requested in the message
    for d in detections:
        cls_name = d.get("class_name", "").lower().split(" (")[0].strip()
        if len(cls_name) > 3 and cls_name in msg:
            count = sum(1 for det in detections if cls_name in det.get("class_name", "").lower())
            LAST_RECOMMENDATION_REASON = f"Because I am tracking {count} {cls_name}(s) in the current frame."
            return f"Yes, my analysis detects {count} {cls_name}(s) right now."

    # Fallback Handling for unknown/unclear
    LAST_RECOMMENDATION_REASON = "Because I did not understand the previous command."
    FAILED_ATTEMPTS += 1
    if FAILED_ATTEMPTS > 1:
        return "Try asking something like: current speed or road status."
    
    return "I didn't get that. You can ask about speed, vehicles, road condition, or weather."

@app.post("/chat")
async def chat_with_ai(req: ChatRequest) -> Any:
    """Conversational AI endpoint — answers questions about the road analysis."""
    if not req.message or len(req.message.strip()) == 0:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    response_text = build_chat_response(req.message, req.context)

    # Generate TTS audio for the response
    audio_url = None
    if TTS_ENGINE:
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            None, generate_tts_audio, response_text[:500], "af_heart", 1.0
        )
        if audio_path and os.path.exists(audio_path):
            audio_url = f"/results/tts_cache/{os.path.basename(audio_path)}"

    return {
        "response": response_text,
        "audio_url": audio_url,
    }

# Serve TTS cache files
@app.get("/results/tts_cache/{filename}")
async def get_tts_cache(filename: str) -> Any:
    file_path = os.path.join(TTS_CACHE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse({"error": "Audio not found"}, status_code=404)


if __name__ == "__main__":
    print("Starting Road Safety AI - Intelligent Advisory System")
    print("Access: http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
