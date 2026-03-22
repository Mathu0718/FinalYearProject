import os
import io
import uuid
import json
import asyncio
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
except Exception as e:
    print(f"Error loading models: {e}")

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


def generate_smart_advisory(detections: list, weather: dict | None, model_type: str) -> dict:
    """Combine detections + weather into an intelligent driving advisory."""
    sign_advisories = []
    damage_advisories = []
    speed_limit = None

    for det in detections:
        name = det.get("class_name", "").split(" (Frame")[0].strip()
        conf = det.get("confidence", 0)
        if conf < 0.3:
            continue

        # Extract speed limit if present
        if "speed limit" in name.lower():
            try:
                speed_val = int(''.join(filter(str.isdigit, name)))
                if speed_limit is None or speed_val < speed_limit:
                    speed_limit = speed_val
            except:
                pass

        # Get advisory text
        if model_type == "road_damage":
            adv = ROAD_DAMAGE_ADVISORIES.get(name)
            if adv and adv not in damage_advisories:
                damage_advisories.append(adv)
        else:
            adv = ADVISORIES.get(name)
            if adv and adv not in sign_advisories:
                sign_advisories.append(adv)

    # Build weather risk assessment
    weather_risk = get_weather_risk(weather) if weather else {"level": "unknown", "factor": 1.0, "warnings": []}

    # Generate combined intelligent advisory
    combined_parts = []

    if speed_limit:
        if weather_risk["level"] == "high":
            adjusted = max(20, speed_limit - 30)
            combined_parts.append(
                f"Speed limit is {speed_limit} km/h, but due to hazardous weather conditions, "
                f"reduce speed to {adjusted} km/h for safety."
            )
        elif weather_risk["level"] == "moderate":
            adjusted = max(20, speed_limit - 15)
            combined_parts.append(
                f"Speed limit is {speed_limit} km/h. Weather advisory: consider reducing to {adjusted} km/h."
            )
        else:
            combined_parts.append(f"Current speed limit is {speed_limit} km/h. Conditions look clear.")

    if damage_advisories:
        combined_parts.append("Road condition alert: " + " ".join(damage_advisories[:3]))
        if speed_limit:
            combined_parts.append("Reduce speed further due to road damage.")

    for w in weather_risk["warnings"][:3]:
        combined_parts.append(w)

    if not combined_parts:
        if sign_advisories:
            combined_parts = sign_advisories[:3]
        else:
            combined_parts.append("Road scan complete. No immediate hazards detected. Drive safely.")

    return {
        "combined_advisory": " ".join(combined_parts),
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

def run_model_on_image(model, img, model_key, annotated, conf_threshold=0.3):
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
        if conf < conf_threshold:
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
            "class_id": cls_id, "source": model_key,
        })
    return detections


def process_image(contents: bytes, filename: str, model_type: str, weather: dict | None = None) -> dict:
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_cv = np.array(img)[..., ::-1].copy()  # RGB to BGR
    annotated = img_cv.copy()

    all_detections = []

    if model_type == 'all_models':
        # Run ALL models
        for mkey in ALL_MODEL_KEYS:
            m = MODELS.get(mkey)
            if m:
                dets = run_model_on_image(m, img, mkey, annotated)
                all_detections.extend(dets)
    else:
        # Run selected model
        model = MODELS.get(model_type)
        if model:
            results = model(img)
            result = results[0]
            annotated = result.plot()
            names = result.names
            for i in range(len(result.boxes)):
                conf = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                all_detections.append({
                    "class_name": names[cls_id], "confidence": round(conf, 3),
                    "class_id": cls_id, "source": model_type,
                })

        # Also run COCO in background
        coco_model = MODELS.get('coco_road')
        if coco_model and model_type != 'coco_road':
            coco_dets = run_model_on_image(coco_model, img, 'coco_road', annotated)
            all_detections.extend(coco_dets)

    # Save annotated image
    annotated_img = Image.fromarray(annotated[..., ::-1])
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    max_frames = 300
    all_detections_history = []

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

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
                    if conf < 0.3:
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
                        "confidence": round(conf, 3), "class_id": cls_id, "source": mkey,
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
                    if cconf < 0.35:
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
                            "confidence": cconf, "class_id": ccls, "source": "coco_road",
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
                            "source": model_type,
                            "representative_url": f"/results/{rep_filename}",
                            "full_frame_url": f"/results/{full_filename}"
                        })

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
) -> Any:

    if model_type not in MODELS and model_type != 'all_models':
        return JSONResponse({"error": f"Model '{model_type}' is not loaded."}, status_code=500)

    # Fetch weather in parallel
    loop = asyncio.get_event_loop()
    weather = await loop.run_in_executor(None, fetch_weather, lat, lon)

    content_type = file.content_type

    if content_type.startswith("image/"):
        contents = await file.read()
        try:
            result = await loop.run_in_executor(None, process_image, contents, file.filename, model_type, weather)
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
):
    """Stream detection events in real-time as video frames are processed."""

    if model_type not in MODELS and model_type != 'all_models':
        return JSONResponse({"error": f"Model '{model_type}' is not loaded."}, status_code=500)

    content_type = file.content_type

    # For images, use regular endpoint logic
    if content_type.startswith("image/"):
        contents = await file.read()
        loop = asyncio.get_event_loop()
        weather = await loop.run_in_executor(None, fetch_weather, lat, lon)
        result = await loop.run_in_executor(None, process_image, contents, file.filename, model_type, weather)
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        max_frames = 300
        all_detections = []
        announced_classes = set()

        yield json.dumps({
            "event": "start",
            "total_frames": min(total_frames, max_frames),
            "weather": weather,
        }) + "\n"

        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break

            frame_new_detections = []
            frame_all_classes = []
            annotated_frame = frame.copy()

            if is_all:
                # Run ALL models on this frame
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
                        if conf < 0.15:
                            continue

                        frame_all_classes.append(label)

                        x1, y1, x2, y2 = box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(annotated_frame, text, (x1 + 2, y1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        if label not in announced_classes:
                            announced_classes.add(label)
                            # Save crop for telemetry thumbnail
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
                            det = {
                                "class_name": label, "confidence": round(conf, 3),
                                "class_id": cls_id, "source": mkey, "frame": frame_count,
                                "representative_url": rep_url, "full_frame_url": full_url,
                            }
                            frame_new_detections.append(det)
                            all_detections.append(det)
            else:
                # Run single selected model
                results = single_model(frame, verbose=False)
                result = results[0]
                annotated_frame = result.plot()
                names = result.names

                # COCO background detection
                if coco_model and model_type != 'coco_road':
                    coco_results = coco_model(frame, verbose=False, classes=COCO_ROAD_CLASSES)
                    coco_result = coco_results[0]
                    coco_names = coco_model.names

                    for j in range(len(coco_result.boxes)):
                        cbox = coco_result.boxes.xyxy[j].cpu().numpy().astype(int)
                        cconf = float(coco_result.boxes.conf[j])
                        ccls = int(coco_result.boxes.cls[j])
                        clabel = coco_names[ccls]
                        if cconf < 0.20:
                            continue

                        frame_all_classes.append(clabel)

                        cx1, cy1, cx2, cy2 = cbox
                        cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                        ctext = f"{clabel} {cconf:.2f}"
                        (tw, th), _ = cv2.getTextSize(ctext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (cx1, cy1 - th - 6), (cx1 + tw + 4, cy1), (255, 255, 0), -1)
                        cv2.putText(annotated_frame, ctext, (cx1 + 2, cy1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        if clabel not in announced_classes:
                            announced_classes.add(clabel)
                            det = {"class_name": clabel, "confidence": round(cconf, 3), "class_id": ccls, "source": "coco_road", "frame": frame_count}
                            frame_new_detections.append(det)
                            all_detections.append(det)

                # Selected model detections
                class_ids = result.boxes.cls.cpu().numpy().tolist()
                confidences = result.boxes.conf.cpu().numpy().tolist()

                for i, cls_id in enumerate(class_ids):
                    c_name = names[int(cls_id)]
                    conf = confidences[i]
                    if conf < 0.15:
                        continue
                    box = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    y1, y2 = max(0, y1), min(height, y2)
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
                            "class_id": int(cls_id), "source": model_type, "frame": frame_count,
                            "representative_url": rep_url, "full_frame_url": full_url,
                        }
                        frame_new_detections.append(det)
                        all_detections.append(det)

            out_writer.write(annotated_frame)

            if frame_new_detections:
                yield json.dumps({"event": "detection", "frame": frame_count, "detections": frame_new_detections}) + "\n"

            if frame_count % 5 == 0:
                yield json.dumps({"event": "speed_sync", "frame": frame_count, "classes": frame_all_classes}) + "\n"

            if frame_count % 20 == 0:
                yield json.dumps({"event": "progress", "frame": frame_count, "total": min(total_frames, max_frames)}) + "\n"

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


if __name__ == "__main__":
    print("Starting Road Safety AI - Intelligent Advisory System")
    print("Access: http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
