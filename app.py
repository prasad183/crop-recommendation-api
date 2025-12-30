from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from pymongo import MongoClient
from fuzzywuzzy import process

# -------------------- APP INIT --------------------
app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

model = joblib.load(MODEL_PATH)

# -------------------- MONGODB CONNECTION --------------------
MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client["cropdb"]
mandi_collection = db["mandi_demand"]

# -------------------- UTILITY FUNCTIONS --------------------
def predict_suitability(inputs):
    """
    inputs = [N, P, K, temperature, humidity, ph, rainfall]
    """
    input_array = np.array(inputs).reshape(1, -1)

    probabilities = model.predict_proba(input_array)[0]
    crops = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:5]

    results = []
    for i in top_indices:
        score = round(float(probabilities[i]), 2)
        results.append({
            "crop": crops[i],
            "suitability_score": score,
            "expected_yield": "20-35 quintals/acre",
            "risk_level": "Low" if score > 0.75 else "Medium" if score > 0.4 else "High"
        })

    return results


def get_best_mandi(city):
    if not city:
        return None

    city = city.lower().strip()

    mandis = mandi_collection.distinct("area")
    if not mandis:
        return None

    best_match, score = process.extractOne(city, mandis)
    return best_match if score > 70 else None


def get_high_demand_crops(mandi):
    if not mandi:
        return []

    docs = mandi_collection.find(
        {"area": mandi, "demand": "High"},
        {"_id": 0, "crop": 1}
    )

    return [d["crop"] for d in docs]

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def home():
    return "Crop Recommendation API is Running!"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        city = data.get("city", "").strip()

        inputs = [
            float(data.get("N", 0)),
            float(data.get("P", 0)),
            float(data.get("K", 0)),
            float(data.get("temperature", 25)),
            float(data.get("humidity", 70)),
            float(data.get("ph", 6.5)),
            float(data.get("rainfall", 200))
        ]

        recommended_crops = predict_suitability(inputs)

        mandi = get_best_mandi(city)
        high_demand_crops = get_high_demand_crops(mandi)

        for crop in recommended_crops:
            if crop["crop"] in high_demand_crops:
                crop["market_demand"] = "High"
                crop["suitability_score"] = min(
                    1.0, crop["suitability_score"] + 0.25
                )
            else:
                crop["market_demand"] = "Low"

        recommended_crops.sort(
            key=lambda x: x["suitability_score"], reverse=True
        )

        response = {
            "city": city,
            "matched_mandi": mandi if mandi else "No close match found",
            "high_demand_crops_in_city": [
                {"crop": c, "market_demand": "High"} for c in high_demand_crops
            ],
            "recommended_crops": recommended_crops
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e)
        }), 500


# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)
