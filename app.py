from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import requests
import joblib
from dotenv import load_dotenv
from datetime import datetime
import random
import torch
from threading import Lock

# -------------------- SETUP --------------------
app = Flask(__name__)
CORS(app)

# Health-check route (for browser test)
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask backend running successfully!"

# Utility: safer joblib loader
def safe_load(path, name):
    try:
        print(f"[LOAD] Loading {name} from {path} ...")
        obj = joblib.load(path)
        print(f"[OK] {name} loaded successfully.")
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        raise

# -------------------- MODEL LOADING --------------------
model_path = os.path.join(os.path.dirname(__file__), "model", "crop_recommendation_model.pkl")
crop_rec_model = safe_load(model_path, "Crop Recommendation Model")

model_path2 = os.path.join(os.path.dirname(__file__), "model", "fertilizer_recommender.pkl")
fert_artifacts = safe_load(model_path2, "Fertilizer Recommender Artifacts")
fert_rec_model = fert_artifacts["model"]
le_soil = fert_artifacts["le_soil"]
le_crop = fert_artifacts["le_crop"]
le_fert = fert_artifacts["le_fert"]

# -------------------- ENV + WEATHER SETUP --------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# -------------------- ROUTES --------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print("[PREDICT] Incoming data:", data)

    input_data = data.get("input")
    if not input_data or len(input_data) != 7:
        return jsonify({"error": "Invalid input data, please send 7 features"}), 400

    features = pd.DataFrame([input_data], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    prediction = crop_rec_model.predict(features)
    print("[PREDICT] Crop Prediction:", prediction)

    return jsonify({"prediction": prediction[0]})


# -------------------- WEATHER ADVISORY --------------------
def get_weather_advisory(city):
    advisory = {"city": city, "current": {}, "forecast": []}
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    try:
        response = requests.get(CURRENT_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        weather = data["weather"][0]["main"]
        alerts = []

        if temp > 35:
            alerts.append("🔥 High temperature alert! Irrigate crops to avoid heat stress.")
        if temp < 10:
            alerts.append("❄️ Low temperature alert! Risk of frost damage.")
        if humidity > 85:
            alerts.append("🌧️ High humidity alert! Possible risk of fungal diseases.")
        if "Rain" in weather:
            alerts.append("☔ Rain expected! Avoid excess irrigation today.")
        if not alerts:
            alerts.append("✅ Weather looks favorable for crops today.")

        advisory["current"] = {
            "temperature": temp,
            "humidity": humidity,
            "condition": weather,
            "alerts": alerts,
        }

    except requests.exceptions.RequestException as e:
        advisory["current"] = {"error": f"Weather API error: {e}"}
        return advisory

    try:
        forecast_res = requests.get(FORECAST_URL, params=params, timeout=10)
        forecast_res.raise_for_status()
        data = forecast_res.json()
        forecasts = []

        for entry in data["list"]:
            date = datetime.fromtimestamp(entry["dt"]).date()
            temp = entry["main"]["temp"]
            humidity = entry["main"]["humidity"]
            weather = entry["weather"][0]["main"]
            forecasts.append([date, temp, humidity, weather])

        df = pd.DataFrame(forecasts, columns=["date", "temp", "humidity", "weather"])
        daily = df.groupby("date").agg({
            "temp": ["mean", "max", "min"],
            "humidity": "mean",
            "weather": lambda x: x.mode()[0],
        }).reset_index()

        for _, row in daily.iterrows():
            date = row["date"]
            avg_temp = row[("temp", "mean")]
            max_temp = row[("temp", "max")]
            min_temp = row[("temp", "min")]
            humidity = row[("humidity", "mean")]
            weather = row[("weather", "<lambda>")]

            msg = f"📅 {date}: Avg Temp {avg_temp:.1f}°C, Humidity {humidity:.0f}%, Condition {weather}."
            if max_temp > 35:
                msg += " 🔥 Heatwave risk – irrigate crops."
            if min_temp < 10:
                msg += " ❄️ Low temperature risk – protect seedlings."
            if humidity > 85:
                msg += " 🌧️ High humidity – possible fungal outbreaks."
            if "Rain" in weather:
                msg += " ☔ Rain – avoid fertilizer application."

            advisory["forecast"].append(msg)

    except requests.exceptions.RequestException as e:
        advisory["forecast"] = [f"Forecast API error: {e}"]

    return advisory


@app.route("/weather", methods=["GET"])
def weather_route():
    city = request.args.get("city")
    print("[WEATHER] City received:", city)
    if not city:
        return jsonify({"error": "Please provide a city parameter"}), 400

    advisory = get_weather_advisory(city)
    print("[WEATHER] Advisory prepared.")
    return jsonify(advisory)
@app.route("/market_price", methods=["POST"])
def market_price():
    import datetime
    from flask import request, jsonify
    import os, requests

    API_KEY = os.getenv("MANDI_PRICE_API_KEY")
    BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    # ✅ Extract user input from frontend POST
    data = request.get_json()
    commodity = data.get("commodity")
    location = data.get("location")
    print(data)
    if not commodity or not location:
        return jsonify({"error": "Please provide both 'commodity' and 'location'."}), 400

    # 🧠 Smartly interpret location
    # It could match either district or state in the dataset
    state = None
    district = location

    def get_mandi_price(commodity, state=None, district=None, limit=100):
        """
        Fetch latest mandi price with fallback:
        1. commodity + state + district
        2. commodity + state
        3. commodity only
        """
        def fetch(params):
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            return data.get("records", [])

        # Try with full filters
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": limit,
            "filters[commodity]": commodity
        }
        if state:
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district

        records = fetch(params)

        # If no data, try with less filters
        if not records and district:
            params.pop("filters[district]", None)
            records = fetch(params)

        if not records and state:
            params.pop("filters[state]", None)
            records = fetch(params)

        if not records:
            return {"error": f"No data found for {commodity}"}

        # Sort by latest date
        records = sorted(
            records,
            key=lambda x: datetime.datetime.strptime(x["arrival_date"], "%d/%m/%Y"),
            reverse=True
        )

        latest = records[0]
        return {
            "commodity": latest["commodity"],
            "state": latest["state"],
            "district": latest["district"],
            "market": latest["market"],
            "date": latest["arrival_date"],
            "min_price": int(latest["min_price"]),
            "max_price": int(latest["max_price"]),
            "modal_price": int(latest["modal_price"])
        }

    # 🧩 Get latest market data
    result = get_mandi_price(commodity, district=location)

    if "error" in result:
        return jsonify(result), 404

    # 💬 Convert to human-friendly prediction
    prediction = (
        f"📈 Market Update for {result['commodity']}:\n"
        f"📍 Location: {result['market']}, {result['district']}, {result['state']}\n"
        f"📅 Date: {result['date']}\n"
        f"💰 Minimum Price: ₹{result['min_price']} per quintal\n"
        f"💰 Maximum Price: ₹{result['max_price']} per quintal\n"
        f"📊 Modal Price: ₹{result['modal_price']} per quintal\n"
        "This should help you plan your buying or selling effectively! 🧑‍🌾"
    )

    # ✅ Send both raw data + formatted prediction
    return jsonify({
        "message": prediction,
        "data": result
    })

# -------------------- FERTILIZER PREDICTION --------------------
@app.route("/predictFert", methods=["POST"])
def predict_fert():
    try:
        data = request.json.get("input")
        print("[FERT] Incoming data:", data)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        nitrogen = float(data.get("nitrogen"))
        phosphorus = float(data.get("phosphorus"))
        potassium = float(data.get("potassium"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        rainfall = float(data.get("rainfall"))
        soil_type = data.get("soil_type")
        crop_type = data.get("crop_type")

        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        features = np.array([[temperature, humidity, rainfall, soil_encoded, crop_encoded, nitrogen, phosphorus, potassium]])
        pred_idx = fert_rec_model.predict(features)[0]
        fertilizer_pred = le_fert.inverse_transform([pred_idx])[0]

        print("[FERT] Prediction:", fertilizer_pred)
        return jsonify({"prediction": fertilizer_pred})

    except Exception as e:
        print("[ERROR] Exception in /predictFert route:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------- CHATBOT --------------------
generation_lock = Lock()
user_sessions = {}

required_fields = {
    "crop_recommendation": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    "fertilizer_suggestion": ["N", "P", "K", "humidity", "temperature", "rainfall", "soil_type", "crop_type"],
    "market_price_query": ["commodity", "location"],
    "weather_query": ["location"],
}
field_types = {
    "N": "number",
    "P": "number",
    "K": "number",
    "temperature": "number",
    "humidity": "number",
    "ph": "number",
    "rainfall": "number",
    "soil_type": "word",
    "crop_type": "word",
    "location": "word",
    "commodity": "word",
    "state": "word",
    "district": "word"
}
intent_keywords = {
    "fertilizer_suggestion": [
        "fertilizer", "fertiliser", "fertilizer recommendation", "fertilizer suggestion"
    ],
    "crop_recommendation": [
        "crop", "crops", "crop recommendation", "crop suggest", "which crop", "grow crop"
    ],
    "pest_detection": [
        "pest", "insect", "disease", "pest detection", "bug"
    ],
    "weather_query": [
        "weather", "temperature", "rain", "climate", "forecast"
    ],
    "market_price_query": [
        "market", "price", "mandi", "market rate", "commodity price", "crop rate"
    ]
}

valid_crops = [
    'Rice', 'Wheat', 'Tobacco', 'Sugarcane', 'Pulses', 'Pomegranate', 'Paddy',
    'Oil seeds', 'Millets', 'Maize', 'Ground nuts', 'Cotton', 'Coffee',
    'Watermelon', 'Barley', 'Kidneybeans', 'Orange'
]

valid_soils = ['Sandy', 'Loamy', 'Clayey', 'Red', 'Black']


from rapidfuzz import process, fuzz

def detect_intent(msg):
    msg = msg.lower().strip()
    
    best_intent = None
    best_score = 0

    for intent, keywords in intent_keywords.items():
        match, score, _ = process.extractOne(
            msg, keywords, scorer=fuzz.token_set_ratio
        )
        if score > best_score:
            best_intent = intent
            best_score = score

    # Optional threshold to ignore unrelated text
    return best_intent if best_score >= 60 else None

from rapidfuzz import process, fuzz
def validate_numeric(value, field_name, min_val=0, max_val=1000):
    try:
        num = float(value)
        if num < min_val or num > max_val:
            return False, f"{field_name} should be between {min_val} and {max_val}."
        return True, None
    except ValueError:
        return False, f"Please enter a valid number for {field_name}."
import difflib

def validate_and_correct_crop(crop):
    crop = crop.strip().lower()
    valid_crops = [c.lower() for c in le_crop.classes_]
    
    close_match = difflib.get_close_matches(crop, valid_crops, n=1, cutoff=0.7)
    if close_match:
        corrected_crop = le_crop.classes_[valid_crops.index(close_match[0])]
        return True, corrected_crop
    else:
        return False, f"❌ '{crop}' not recognized. Try one of: {', '.join(le_crop.classes_)}."

def validate_and_correct_soil(soil_input):
    soil_input = str(soil_input).strip().title()
    match, score, _ = process.extractOne(soil_input, valid_soils, scorer=fuzz.token_set_ratio)
    if score >= 70 and match in valid_soils:
        return True, match
    return False, f"⚠️ '{soil_input}' is not a valid soil type. Please choose from: {', '.join(valid_soils)}."


def validate_input(field, value):
    expected_type = field_types.get(field)
    if not expected_type:
        return True, None  # no rule, so accept

    if expected_type == "number":
        try:
            float(value)
            return True, None
        except ValueError:
            return False, f"Please enter a valid number for {field} (e.g., 23.5)"
    
    elif expected_type == "word":
        # Ensure no digits or symbols in text
        if any(ch.isdigit() for ch in value):
            return False, f"Please enter a valid word for {field} (not a number)"
        return True, None

    return True, None


def handle_smalltalk(msg: str):
    lowered = msg.lower().strip()

    greetings = ["hi", "hello", "hey", "hii", "heyy"]
    how_are_you = ["how are you", "how r u", "how's it going"]

    # ✅ Only respond if the entire message is smalltalk
    if lowered in greetings:
        return "Hello there! 👋 How can I assist you today?"
    if lowered in how_are_you:
        return "I'm just a bot, but I'm doing great! 😊 How can I help you on the farm today?"
    if lowered in ["thanks", "thank you"]:
        return "You're welcome! 🌾 Anything else I can help with?"

    # If message includes a greeting but also other words → ignore (so intent can process)
    if any(greet in lowered for greet in greetings) and len(lowered.split()) <= 3:
        return "Hi there! 👋 What would you like help with today?"

    return None


# def detect_intent(msg):
    lowered = msg.lower()

    # Fertilizer recommendation
    if "fertilizer" in lowered and "recommendation" in lowered:
        return "fertilizer_suggestion"

    # Crop recommendation
    if "crop" in lowered and "recommendation" in lowered:
        return "crop_recommendation"

    # Pest detection
    if "pest" in lowered and ("detect" in lowered or "detection" in lowered):
        return "pest_detection"

    # Weather query
    if "weather" in lowered:
        return "weather_query"

    # Market or mandi price query
    if (("market" in lowered or "mandi" in lowered) and "price" in lowered):
        return "market_price_query"

    return None


def get_pest_detection(data): return "No pests detected in your field 🌱"
@app.route("/chatbot", methods=["POST"])
def chatbot():
    req_json = request.json
    print("[CHATBOT] Incoming JSON:", req_json)

    user_msg = req_json.get("message", "").strip()
    user_id = req_json.get("user_id", "default")

    if not user_msg:
        return jsonify({"bot": "Please enter a message to continue 😊"})

    # Initialize user session if new
    if user_id not in user_sessions:
        user_sessions[user_id] = {"intent": None, "data": {}, "awaiting": None}
    session = user_sessions[user_id]

    # 🗣 Smalltalk handler
    smalltalk_reply = handle_smalltalk(user_msg)
    if smalltalk_reply:
        return jsonify({"bot": smalltalk_reply})

    # 🎯 Detect or reuse intent
    intent = session["intent"]
    user_intent = req_json.get("intent")

    # ✅ Always define expected_fields at start
    expected_fields = None

    # If user selected from options manually
    if user_intent:
        session["intent"] = user_intent
        session["data"] = {}
        session["awaiting"] = None
        expected_fields = required_fields.get(user_intent, [])

        if expected_fields:
            session["awaiting"] = expected_fields[0]
            return jsonify({
                "bot": f"Sure! You want help with {user_intent.replace('_', ' ')}. Please provide {session['awaiting']}."
            })

    # If no intent yet and user typed something random
    if not intent:
        return jsonify({
            "bot": "I can help you with a variety of tasks 🌾 Please tell me what you'd like to do today!",
            "options": [
                {"label": "🌾 Crop Recommendation", "value": "crop_recommendation"},
                {"label": "💧 Fertilizer Suggestion", "value": "fertilizer_suggestion"},
                {"label": "🐛 Pest Detection", "value": "pest_detection"},
                {"label": "🌤️ Weather Info", "value": "weather_query"},
                {"label": "📈 Market Prices", "value": "market_price_query"}
            ]
        })
    intent = session["intent"]


    expected_fields = required_fields.get(intent, [])

    # Ask for first field if none collected yet
    if session["awaiting"] is None and expected_fields:
        session["awaiting"] = expected_fields[0]
        return jsonify({"bot": f"Sure! You want help with {intent.replace('_', ' ')}. Please provide {session['awaiting']}."})

    # ✅ Validation section
    if session["awaiting"]:
        field = session["awaiting"]

        # Special handling for soil and crop fields
        if field == "crop_type":
            is_valid, result = validate_and_correct_crop(user_msg)
            if not is_valid:
                return jsonify({"bot": result})
            user_msg = result

        elif field == "soil_type":
            is_valid, result = validate_and_correct_soil(user_msg)
            if not is_valid:
                return jsonify({"bot": result})
            user_msg = result

        else:
            is_valid, error_msg = validate_input(field, user_msg)
            if not is_valid:
                return jsonify({"bot": error_msg})

        # ✅ Store valid input
        session["data"][field] = user_msg
        session["awaiting"] = None

    # Check remaining fields
    missing_fields = [f for f in expected_fields if f not in session["data"]]
    if missing_fields:
        session["awaiting"] = missing_fields[0]
        return jsonify({"bot": f"Please provide {session['awaiting']}."})

    # -------------------- PROCESS INTENT --------------------
    data = session["data"]
    intent = session["intent"]

    if intent == "crop_recommendation":
        features = pd.DataFrame([data], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        model_pred = crop_rec_model.predict(features)
        prediction = f"Based on your inputs, we recommend growing 🌾 {model_pred[0]}!"

    elif intent == "fertilizer_suggestion":
        nitrogen = float(data.get("N"))
        phosphorus = float(data.get("P"))
        potassium = float(data.get("K"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        rainfall = float(data.get("rainfall"))
        soil_type = data.get("soil_type").capitalize()
        crop_type = data.get("crop_type").capitalize()

        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        features = np.array([[temperature, humidity, rainfall, soil_encoded, crop_encoded, nitrogen, phosphorus, potassium]])
        pred_idx = fert_rec_model.predict(features)[0]
        fertilizer_pred = le_fert.inverse_transform([pred_idx])[0]

        prediction = f"Given your values, the suggested fertilizer is 🌿 {fertilizer_pred}."

    elif intent == "pest_detection":
      prediction="Upload your crop image here to detect pests instantly:<br><br>🐛 <b>Pest Detection</b><br><br>👉 <a href='https://krishi-gram.vercel.app/pest_detection' target='_blank' style='color:blue;text-decoration:underline;'>Click here to open Pest Detection</a><br><br>It will analyze the image using the trained model! 📸🌱'"
        



    elif intent == "weather_query":
        city = data.get("location", "your area")
        advisory = get_weather_advisory(city)

        if "error" in advisory["current"]:
            prediction = f"⚠️ Weather API error: {advisory['current']['error']}"
        else:
            current_msg = (
                f"🌤️ Current weather in {city}: {advisory['current']['temperature']:.1f}°C, "
                f"Humidity: {advisory['current']['humidity']}%, Condition: {advisory['current']['condition']}. "
                + " ".join(advisory["current"]["alerts"])
            )
            prediction = f"{current_msg} Hope this helps you plan your day on the farm! 🌾"

    elif intent == "market_price_query":
        API_KEY = os.getenv("MANDI_PRICE_API_KEY")
        BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

        commodity = data.get("commodity", "Wheat")
        state = data.get("state")
        district = data.get("district")
        limit = 100

        def get_mandi_price(commodity, state=None, district=None, limit=100):
            def fetch(params):
                response = requests.get(BASE_URL, params=params, timeout=10)
                data = response.json()
                return data.get("records", [])

            params = {
                "api-key": API_KEY,
                "format": "json",
                "limit": limit,
                "filters[commodity]": commodity
            }

            if state:
                params["filters[state]"] = state
            if district:
                params["filters[district]"] = district

            records = fetch(params)
            if district:
                records = [r for r in records if r.get("district", "").lower() == district.lower()]
            if not records and district:
                params.pop("filters[district]", None)
                records = fetch(params)
            if not records and state:
                params.pop("filters[state]", None)
                records = fetch(params)
            if not records:
                return None

            records = sorted(records, key=lambda x: datetime.strptime(x["arrival_date"], "%d/%m/%Y"), reverse=True)
            return records[0]

        latest = get_mandi_price(commodity, state, district, limit)
        if latest:
            prediction = (
                f"📈 Market Update for {latest['commodity']}:\n"
                f"Date: {latest['arrival_date']}\n"
                f"Minimum Price: ₹{int(latest['min_price'])} per quintal\n"
                f"Maximum Price: ₹{int(latest['max_price'])} per quintal\n"
                f"Modal Price: ₹{int(latest['modal_price'])} per quintal\n"
                "This should help you plan your buying or selling effectively! 🧑‍🌾"
            )
        else:
            prediction = f"⚠️ Sorry, no market data found for {commodity}."

    else:
        prediction = "I'm not sure how to help with that yet 🤔"

    # ✅ Reset after completing task
    session["intent"] = None
    session["data"] = {}
    session["awaiting"] = None

    return jsonify({"bot": prediction + " Would you like to check anything else?"})

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("🚀 Starting Flask backend on port 8080...")
    app.run(debug=True, port=8080)

