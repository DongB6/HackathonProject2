from flask import Flask, render_template, request, jsonify
import re
import json

app = Flask(__name__)

# Medical terminology dictionary for demo purposes
# In production, you'd use a proper medical API or database
MEDICAL_TERMS = {
    "myocardial infarction": {
        "simple": "Heart attack - when blood flow to part of your heart muscle gets blocked, usually by a blood clot.",
        "symptoms": "Chest pain, shortness of breath, nausea, sweating",
        "causes": "Blocked coronary arteries, usually from plaque buildup",
        "prevention": "Regular exercise, healthy diet, don't smoke, manage stress"
    },
    "hypertension": {
        "simple": "High blood pressure - when the force of blood against your artery walls is too high.",
        "symptoms": "Often no symptoms (silent killer), but can cause headaches, dizziness",
        "causes": "Poor diet, lack of exercise, stress, genetics, age",
        "prevention": "Low-salt diet, regular exercise, maintain healthy weight, limit alcohol"
    },
    "diabetes mellitus": {
        "simple": "Diabetes - when your body can't properly control blood sugar levels.",
        "symptoms": "Increased thirst, frequent urination, fatigue, blurred vision",
        "causes": "Body doesn't make enough insulin or can't use insulin properly",
        "prevention": "Healthy diet, regular exercise, maintain healthy weight"
    },
    "pneumonia": {
        "simple": "Lung infection - when your lungs get infected and inflamed, making it hard to breathe.",
        "symptoms": "Cough, fever, difficulty breathing, chest pain",
        "causes": "Bacteria, viruses, or fungi infecting the lungs",
        "prevention": "Get vaccinated, wash hands frequently, don't smoke, stay healthy"
    },
    "osteoporosis": {
        "simple": "Weak bones - when your bones become thin and brittle, breaking easily.",
        "symptoms": "Usually no symptoms until a bone breaks, back pain, loss of height",
        "causes": "Lack of calcium, vitamin D deficiency, aging, hormonal changes",
        "prevention": "Calcium-rich foods, vitamin D, weight-bearing exercise, avoid smoking"
    },
    "gastroesophageal reflux disease": {
        "simple": "Acid reflux (GERD) - when stomach acid flows back up into your throat, causing heartburn.",
        "symptoms": "Heartburn, chest pain, difficulty swallowing, regurgitation",
        "causes": "Weak lower esophageal sphincter, certain foods, obesity",
        "prevention": "Avoid trigger foods, eat smaller meals, don't lie down after eating"
    }
}

def simplify_medical_text(text):
    """
    Simplified medical text processor
    In production, this would use a proper NLP model like BioBERT
    """
    text_lower = text.lower().strip()
    
    # Check for exact matches first
    for term, info in MEDICAL_TERMS.items():
        if term in text_lower:
            return info
    
    # Check for partial matches or synonyms
    synonyms = {
        "heart attack": "myocardial infarction",
        "high blood pressure": "hypertension",
        "sugar diabetes": "diabetes mellitus",
        "lung infection": "pneumonia",
        "brittle bones": "osteoporosis",
        "heartburn": "gastroesophageal reflux disease",
        "gerd": "gastroesophageal reflux disease",
        "acid reflux": "gastroesophageal reflux disease"
    }
    
    for synonym, medical_term in synonyms.items():
        if synonym in text_lower:
            return MEDICAL_TERMS.get(medical_term, {})
    
    # Default response for unknown terms
    return {
        "simple": "I don't have information about this specific condition yet. Please consult with a healthcare professional for accurate medical information.",
        "symptoms": "Unknown - please consult a doctor",
        "causes": "Unknown - please consult a doctor", 
        "prevention": "General advice: maintain a healthy lifestyle with good diet and exercise"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    medical_text = data.get('text', '')
    difficulty = data.get('difficulty', 'simple')
    
    if not medical_text:
        return jsonify({'error': 'Please enter a medical condition to explain'}), 400
    
    explanation = simplify_medical_text(medical_text)
    
    response = {
        'original': medical_text,
        'explanation': explanation,
        'difficulty': difficulty
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)