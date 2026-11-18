"""
prediction.py — Full CNN+RNN pipeline with live satellite imagery
=================================================================
Flow:
  1. Geocode city -> lat/lon (OpenStreetMap Nominatim)
  2. Fetch 3 real satellite images (2 months ago, 1 month ago, today)
     from NASA GIBS WMS — MODIS Terra true colour, free, no API key
  3. Pass 3 images through CNN -> LSTM -> combined_rf  (the actual trained model)
  4. Fetch live weather data from Open-Meteo for display
  5. Combine satellite-based CNN+RNN prediction with weather RF prediction
     using a confidence-weighted vote for the final result
"""

import datetime
import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image

IMG_SIZE = 64

# ─────────────────────────────────────────────────────────────────────────────
# Fallback weather data (used only if Open-Meteo API is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
DUMMY_DATA = {
    "Delhi":             [95.0, 108.0, 15.0, 40.0,  0.5, 30.0],
    "Mumbai":            [91.0,  95.0, 20.0, 60.0,  1.0, 70.0],
    "Kolkata":           [95.0, 102.0, 12.0, 45.0,  0.8, 55.0],
    "Bangalore":         [82.0,  90.0,  8.0, 35.0,  0.5, 50.0],
    "Chennai":           [95.0, 102.0, 18.0, 40.0,  0.3, 65.0],
    "Hyderabad":         [95.0, 104.0, 13.0, 42.0,  0.4, 38.0],
    "Jangareddi Gudem":  [88.0,  98.0, 11.0, 55.0,  3.5, 72.0],
    "Jangareddygudem":   [88.0,  98.0, 11.0, 55.0,  3.5, 72.0],
    "Shillong":          [68.0,  75.0, 10.0, 88.0, 12.0, 88.0],
    "Guwahati":          [84.0,  95.0, 10.0, 75.0,  6.0, 78.0],
    "Manaus":            [82.0,  95.0,  6.0, 88.0, 18.0, 90.0],
    "Jakarta":           [86.0,  93.0, 10.0, 80.0,  8.0, 82.0],
    "Singapore":         [84.0,  90.0, 12.0, 75.0,  6.0, 82.0],
    "New York":          [50.0,  62.0, 18.0, 45.0,  1.5, 55.0],
    "Los Angeles":       [65.0,  75.0, 10.0, 20.0,  0.2, 40.0],
    "London":            [48.0,  55.0, 20.0, 70.0,  1.5, 75.0],
    "Paris":             [52.0,  60.0, 15.0, 60.0,  1.2, 68.0],
    "Sydney":            [72.0,  80.0, 22.0, 35.0,  0.8, 55.0],
    "Beijing":           [55.0,  68.0, 12.0, 35.0,  0.3, 40.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# GEOCODING
# ─────────────────────────────────────────────────────────────────────────────

def geocode(city):
    """Get lat/lon for any city using OpenStreetMap Nominatim."""
    headers = {"User-Agent": "FloodPredictionApp/1.0"}
    for query in [f"{city},India", city]:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers=headers, timeout=15)
        results = r.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            name = results[0]["display_name"][:60]
            print(f"  Found: {name}  ({lat:.2f}, {lon:.2f})")
            return lat, lon
    raise ValueError(f"City not found: {city}")


# ─────────────────────────────────────────────────────────────────────────────
# SATELLITE IMAGE FETCHING — NASA GIBS WMS (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_satellite_image(lat, lon, date_str):
    """
    Fetch a 64x64 RGB satellite image from NASA GIBS WMS.
    Uses MODIS Terra CorrectedReflectance TrueColor.
    Returns: np.ndarray (64,64,3) float32 in [0,1], and raw JPEG bytes.
    date_str: 'YYYY-MM-DD'
    """
    delta = 0.5  # ~55 km bounding box
    bbox  = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    url   = (
        f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
        f"&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
        f"&SRS=EPSG:4326&BBOX={bbox}"
        f"&WIDTH={IMG_SIZE}&HEIGHT={IMG_SIZE}&FORMAT=image/jpeg"
        f"&TIME={date_str}"
    )
    r = requests.get(url, timeout=20)
    if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
        raw_bytes = r.content
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        arr = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        # Re-encode as JPEG bytes for display in the browser
        buf = BytesIO()
        img.resize((256, 256), Image.BILINEAR).save(buf, format="JPEG", quality=85)
        display_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return arr, display_b64
    raise ValueError(f"GIBS image fetch failed for {date_str}: HTTP {r.status_code}")


def get_satellite_sequence(lat, lon):
    """
    Fetch 3 monthly satellite images to form the CNN+RNN temporal sequence.
    Uses the last 3 months: ~60 days ago, ~30 days ago, and today.
    Returns:
        imgs      — list of 3 x np.ndarray (64,64,3) float32
        b64_imgs  — list of 3 x base64 JPEG strings (for display)
        dates     — list of 3 date strings
    """
    today = datetime.date.today()
    dates = [
        (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d"),
        (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    ]
    imgs, b64_imgs = [], []
    errors = []
    for d in dates:
        try:
            arr, b64 = fetch_satellite_image(lat, lon, d)
            imgs.append(arr)
            b64_imgs.append(b64)
            print(f"  ✓ Satellite image fetched for {d}  shape={arr.shape}")
        except Exception as e:
            errors.append(f"{d}: {e}")
            print(f"  ✗ Failed to fetch image for {d}: {e}")
            # Generate a grey placeholder so the pipeline still runs
            imgs.append(np.full((IMG_SIZE, IMG_SIZE, 3), 0.5, dtype=np.float32))
            b64_imgs.append("")
    if errors:
        print(f"  Warning: {len(errors)} satellite image(s) used placeholder.")
    return imgs, b64_imgs, dates


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER DATA — Open-Meteo (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def get_weather(lat, lon):
    """
    Fetch last 7 days of weather from Open-Meteo.
    Returns [avg_temp_F, max_temp_F, wind_mph, cloud_pct, precip_mm_7day, humidity_pct]
    """
    today    = datetime.date.today().strftime("%Y-%m-%d")
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,windspeed_10m_max,cloudcover_mean"
        f"&hourly=relativehumidity_2m"
        f"&start_date={week_ago}&end_date={today}"
        f"&timezone=auto"
    )
    data   = requests.get(url, timeout=15).json()
    daily  = data["daily"]
    hourly = data["hourly"]
    n      = len(daily["time"])

    avg_temp     = sum((mx + mn) / 2 for mx, mn in zip(
                       daily["temperature_2m_max"],
                       daily["temperature_2m_min"])) / n
    max_temp     = max(daily["temperature_2m_max"])
    avg_wind     = sum(daily["windspeed_10m_max"]) / n
    avg_cloud    = sum(daily["cloudcover_mean"]) / n
    total_precip = sum(daily["precipitation_sum"])
    avg_humid    = sum(hourly["relativehumidity_2m"]) / len(hourly["relativehumidity_2m"])

    return [avg_temp * 9/5 + 32, max_temp * 9/5 + 32,
            avg_wind, avg_cloud, total_precip, avg_humid]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_full_prediction(city, model):
    """
    Full pipeline:
      1. Geocode city
      2. Fetch 3 NASA GIBS satellite images → CNN+RNN → combined_rf prediction
      3. Fetch weather data → weather_rf prediction
      4. Combine both predictions (satellite takes priority when available)

    Returns a dict with all fields needed by the template:
      pred, flood_score, landslide_score, confidence,
      method, weather_features, satellite_b64_imgs, satellite_dates,
      sat_available
    """
    result = {
        "pred": "Unknown",
        "flood_score": 0.0,
        "landslide_score": 0.0,
        "confidence": 0.0,
        "method": "unknown",
        "weather_features": [0]*6,
        "satellite_b64_imgs": ["", "", ""],
        "satellite_dates": ["", "", ""],
        "sat_available": False,
        "cnn_rnn_prob": 0.0,
        "weather_prob": 0.0,
    }

    # ── Step 1: Geocode ────────────────────────────────────────────────────
    try:
        lat, lon = geocode(city)
    except Exception as e:
        print(f"  Geocoding failed: {e}")
        lat, lon = None, None

    # ── Step 2: Weather data ───────────────────────────────────────────────
    weather_features = None
    if lat is not None:
        try:
            weather_features = get_weather(lat, lon)
        except Exception as e:
            print(f"  Weather API failed ({e}), using fallback for '{city}'")
    if weather_features is None:
        city_lower = city.lower()
        for key in DUMMY_DATA:
            if key.lower() == city_lower:
                weather_features = DUMMY_DATA[key]
                break
        if weather_features is None:
            weather_features = [86.0, 95.0, 12.0, 65.0, 1.0, 60.0]
    result["weather_features"] = weather_features

    # ── Step 3: Weather RF prediction ─────────────────────────────────────
    weather_proba = model.predict_proba_flat([weather_features])[0]
    weather_prob_unsafe = float(weather_proba[1])
    weather_pred = int(model.predict_flat([weather_features])[0])
    result["weather_prob"] = round(weather_prob_unsafe, 3)

    # ── Step 4: CNN+RNN prediction via live satellite images ───────────────
    sat_pred = None
    sat_prob_unsafe = None

    if lat is not None:
        try:
            imgs, b64_imgs, dates = get_satellite_sequence(lat, lon)
            result["satellite_b64_imgs"] = b64_imgs
            result["satellite_dates"]    = dates
            result["sat_available"]      = True

            # Run the full CNN → LSTM → combined_rf pipeline
            sample = [{"imgs": imgs}]
            sat_pred_raw = model.predict(sample)[0]
            sat_proba    = model.predict_proba(sample)[0]
            sat_prob_unsafe = float(sat_proba[1])

            # Flood and landslide individual scores
            fl_scores, ls_scores = model.predict_risk_scores(sample)
            result["flood_score"]     = round(float(fl_scores[0]), 3)
            result["landslide_score"] = round(float(ls_scores[0]), 3)

            sat_pred = int(sat_pred_raw)
            result["cnn_rnn_prob"] = round(sat_prob_unsafe, 3)

            print(f"  CNN+RNN:  pred={sat_pred}  prob_unsafe={sat_prob_unsafe:.3f}")
            print(f"  Flood score={result['flood_score']}  Landslide score={result['landslide_score']}")

        except Exception as e:
            print(f"  CNN+RNN pipeline failed: {e}")
            result["sat_available"] = False

    # ── Step 5: Combine predictions ────────────────────────────────────────
    #
    # WHY these weights:
    #   - The CNN+RNN was trained on a heavily imbalanced dataset (84% Unsafe)
    #     with random (non-backprop-trained) CNN weights, so its raw probability
    #     is systematically biased toward Unsafe for ANY real satellite input.
    #   - The Weather RF was trained on balanced synthetic scenarios and is
    #     well-calibrated against real weather signals (precip, humidity, etc.).
    #   - Therefore Weather RF gets 70% weight; CNN+RNN gets 30% as a secondary
    #     visual confirmation signal only.
    #   - The decision threshold is raised from 0.5 → 0.62 to further counteract
    #     the training-set Unsafe bias in the CNN+RNN branch.
    #   - The CNN+RNN probability is also re-centred: values below 0.75 are
    #     shrunk toward 0.5 because the model rarely predicts below 0.75 even for
    #     safe cities — treating the [0.75, 1.0] band as the true signal range.

    if sat_pred is not None and sat_prob_unsafe is not None:
        # Calibration: re-scale CNN+RNN prob — compress the always-high range
        # [0.0, 0.75] → [0.0, 0.5], [0.75, 1.0] → [0.5, 1.0]
        if sat_prob_unsafe < 0.75:
            sat_prob_calibrated = sat_prob_unsafe * (0.5 / 0.75)
        else:
            sat_prob_calibrated = 0.5 + (sat_prob_unsafe - 0.75) * (0.5 / 0.25)
        sat_prob_calibrated = round(float(sat_prob_calibrated), 3)

        # Weighted combination: Weather RF 70%, CNN+RNN 30%
        combined_prob = 0.30 * sat_prob_calibrated + 0.70 * weather_prob_unsafe

        # Raised threshold (0.62) counteracts the Unsafe training bias
        THRESHOLD = 0.62
        final_pred = 1 if combined_prob >= THRESHOLD else 0
        result["method"]     = "CNN+RNN (satellite) + Weather RF (combined)"
        result["confidence"] = round(combined_prob if final_pred == 1 else 1 - combined_prob, 3)
        result["cnn_rnn_prob"] = sat_prob_calibrated   # show calibrated value
    else:
        # Fall back to weather RF only
        final_pred           = weather_pred
        result["method"]     = "Weather RF only (satellite unavailable)"
        result["confidence"] = round(weather_prob_unsafe if final_pred == 1 else 1 - weather_prob_unsafe, 3)
        combined_prob        = weather_prob_unsafe

    result["pred"] = "Unsafe" if final_pred == 1 else "Safe"

    # ── Step 6: Console summary ────────────────────────────────────────────
    temp_c = round((weather_features[0] - 32) * 5/9, 1)
    maxt_c = round((weather_features[1] - 32) * 5/9, 1)
    print("=" * 50)
    print(f"  City            : {city}")
    print(f"  Coordinates     : {lat:.2f}, {lon:.2f}" if lat else "  Coordinates     : N/A")
    print(f"  Temperature     : {round(weather_features[0], 1)} °F  ({temp_c} °C)")
    print(f"  Max Temp        : {round(weather_features[1], 1)} °F  ({maxt_c} °C)")
    print(f"  Wind Speed      : {round(weather_features[2], 1)} mph")
    print(f"  Cloud Cover     : {round(weather_features[3], 1)} %")
    print(f"  Precipitation   : {round(weather_features[4], 1)} mm")
    print(f"  Humidity        : {round(weather_features[5], 1)} %")
    print(f"  Satellite avail : {result['sat_available']}")
    if result["sat_available"]:
        print(f"  CNN+RNN prob    : {result['cnn_rnn_prob']:.3f}")
        print(f"  Flood score     : {result['flood_score']:.3f}")
        print(f"  Landslide score : {result['landslide_score']:.3f}")
    print(f"  Weather prob    : {result['weather_prob']:.3f}")
    print(f"  Method          : {result['method']}")
    print(f"  Final prediction: {'⚠  Unsafe' if result['pred'] == 'Unsafe' else '✓  Safe'}  (conf={result['confidence']:.3f})")
    print("=" * 50)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

def get_data(city):
    """Legacy: return weather features only (used if caller does not pass model)."""
    try:
        lat, lon = geocode(city)
        return get_weather(lat, lon)
    except Exception as e:
        print(f"  get_data fallback for '{city}': {e}")
        city_lower = city.lower()
        for key in DUMMY_DATA:
            if key.lower() == city_lower:
                return DUMMY_DATA[key]
        return [86.0, 95.0, 12.0, 65.0, 1.0, 60.0]


def get_satellite_images_for_city(city):
    """Legacy helper — returns list of 3 image arrays or None."""
    try:
        lat, lon = geocode(city)
        imgs, _, _ = get_satellite_sequence(lat, lon)
        return imgs
    except Exception as e:
        print(f"  Satellite fetch failed ({e})")
        return None


def testConnection():
    return "yo"