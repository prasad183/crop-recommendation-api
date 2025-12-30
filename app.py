from flask import Flask, request, jsonify
import joblib
import numpy as np
from fuzzywuzzy import process
import os
from pymongo import MongoClient

app = Flask(__name__)

# ================== LOAD ML MODEL ==================
MODEL_PATH = r'C:\Users\bhyri\Desktop\crop_recommendation_api\crop_model.pkl'
model = joblib.load(MODEL_PATH)

# ================== MONGODB CONNECTION ==================
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("❌ MONGO_URI environment variable not set")

client = MongoClient(MONGO_URI)
db = client["cropdb"]

# ================== HELPER FUNCTIONS ==================

# Get best mandi name using fuzzy match from DB
def get_best_mandi(city):
    if not city:
        return None

    city_lower = city.lower().strip()

    mandis = db.mandi_demand.distinct("mandi")
    if not mandis:
        return None

    best_match, score = process.extractOne(city_lower, mandis)
    return best_match if score > 75 else None


# Fetch high-demand crops from MongoDB
def get_high_demand_crops_from_db(mandi):
    cursor = db.mandi_demand.find(
        {"mandi": mandi},
        {"_id": 0, "crop": 1, "arrivals_tonnes": 1}
    ).sort("arrivals_tonnes", -1).limit(5)

    return list(cursor)


# ML prediction (UNCHANGED)
def predict_suitability(inputs):
    input_array = np.array([inputs]).reshape(1, -1)
    probabilities = model.predict_proba(input_array)[0]
    crop_classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:5]
    results = []

    for i in top_indices:
        crop = crop_classes[i]
        score = round(probabilities[i], 2)
        results.append({
            "crop": crop,
            "suitability_score": score,
            "expected_yield": "20-35 quintals/acre",
            "risk_level": "Low" if score > 0.8 else "Medium" if score > 0.5 else "High"
        })

    return results


# ================== API ROUTES ==================

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    city = data.get("city", "").strip()
    N = float(data.get("N", 0))
    P = float(data.get("P", 0))
    K = float(data.get("K", 0))
    temperature = float(data.get("temperature", 25))
    humidity = float(data.get("humidity", 70))
    ph = float(data.get("ph", 6.5))
    rainfall = float(data.get("rainfall", 200))

    inputs = [N, P, K, temperature, humidity, ph, rainfall]
    recommended_crops = predict_suitability(inputs)

    # ---- MongoDB mandi demand ----
    mandi = get_best_mandi(city)
    high_demand_list = get_high_demand_crops_from_db(mandi) if mandi else []
    high_demand_crop_names = [item["crop"] for item in high_demand_list]

    # Boost ML scores based on mandi demand
    for rec in recommended_crops:
        if rec["crop"] in high_demand_crop_names:
            rec["suitability_score"] = min(1.0, rec["suitability_score"] + 0.25)
            rec["market_demand"] = "High"
        else:
            rec["market_demand"] = "Low"

    recommended_crops.sort(key=lambda x: x["suitability_score"], reverse=True)

    clean_high_demand = [
        {"crop": item["crop"], "market_demand": "High"}
        for item in high_demand_list
    ]

    response = {
        "city": city,
        "matched_mandi": mandi or "No close match found",
        "recommended_crops": recommended_crops,
        "high_demand_crops_in_city": clean_high_demand
    }

    return jsonify(response)


@app.route("/")
def home():
    return (
        "<h1>Crop Recommendation API is Running!</h1>"
        "<p>Send POST request to <b>/recommend</b> with JSON body.</p>"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
