from flask import Flask, render_template, request, jsonify
import logging
import torch
import os 
from transformers import pipeline
import json
import re

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the model pipeline
simplifier_pipeline = None

def initialize_model():
    """Initialize the FLAN-T5 model for medical text simplification"""
    global simplifier_pipeline
    
    try:
        logger.info("Loading FLAN-T5 model...")
        model_name = "./models/med_simplifier"
        
        # Check if model path exists
        if not os.path.exists(model_name):
            logger.warning(f"Model path {model_name} does not exist. Using fallback mode.")
            simplifier_pipeline = None
            return
            
        simplifier_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1,  # Force CPU to avoid CUDA issues
            torch_dtype=torch.float32
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Continuing with fallback dictionary mode")
        simplifier_pipeline = None

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
        "simple": "Bone thinning disease - when your bones become weak and brittle, making them more likely to break.",
        "symptoms": "Back pain, loss of height, stooped posture, bones that break easily",
        "causes": "Low calcium, lack of exercise, aging, hormonal changes, certain medications",
        "prevention": "Get enough calcium and vitamin D, regular weight-bearing exercise, don't smoke, limit alcohol"
    },
    "acid reflux": {
        "simple": "Heartburn condition - when stomach acid flows back up into your food pipe (esophagus), causing burning pain.",
        "symptoms": "Burning chest pain (heartburn), sour taste in mouth, difficulty swallowing, chronic cough",
        "causes": "Weak valve between stomach and esophagus, eating large meals, lying down after eating, certain foods",
        "prevention": "Eat smaller meals, avoid trigger foods (spicy, fatty, acidic), don't lie down after eating, maintain healthy weight"
    },
    "gastroesophageal reflux disease": {
        "simple": "Chronic heartburn condition (GERD) - when stomach acid regularly flows back into your food pipe, causing irritation.",
        "symptoms": "Frequent heartburn, chest pain, difficulty swallowing, chronic cough, sore throat",
        "causes": "Weak lower esophageal sphincter, hiatal hernia, certain foods, obesity",
        "prevention": "Avoid trigger foods, eat smaller meals, maintain healthy weight, elevate head when sleeping"
    }
}

# Enhanced synonym mapping
SYNONYMS = {
    "heart attack": "myocardial infarction",
    "cardiac arrest": "myocardial infarction",
    "high blood pressure": "hypertension",
    "elevated blood pressure": "hypertension",
    "high bp": "hypertension",
    "diabetes": "diabetes mellitus",
    "sugar diabetes": "diabetes mellitus",
    "high blood sugar": "diabetes mellitus",
    "lung infection": "pneumonia",
    "chest infection": "pneumonia",
    "respiratory infection": "pneumonia",
    "bone loss": "osteoporosis",
    "brittle bones": "osteoporosis", 
    "weak bones": "osteoporosis",
    "bone thinning": "osteoporosis",
    "bone disease": "osteoporosis",
    "heartburn": "acid reflux",
    "gerd": "gastroesophageal reflux disease",
    "reflux": "acid reflux",
    "stomach acid": "acid reflux",
    "indigestion": "acid reflux",
    "gastric reflux": "acid reflux"
}

# Load medical terms from JSON file
try:
    with open("output/medical_terms_combined.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.get("terms", []):
        MEDICAL_TERMS[entry["title"].lower()] = {
            "simple": entry["description"],
            "symptoms": entry.get("symptoms", ""),
            "causes": entry.get("causes", ""),
            "prevention": entry.get("prevention", "")
        }
    logger.info("Medical terms loaded from JSON file.")
except Exception as e:
    logger.error(f"Error loading medical terms from JSON: {str(e)}")

def simplify_with_ai_model(text):
    if simplifier_pipeline is None:
        logger.info("Model not loaded, using fallback")
        return None
    try:
        logger.info(f"Running AI model simplification for: {text}")
        result = simplifier_pipeline(text, max_length=256, clean_up_tokenization_spaces=True)
        simplified_text = result[0]["generated_text"]
        logger.info(f"AI output: {simplified_text}")
        return simplified_text
    except Exception as e:
        logger.error(f"Error in AI simplification: {str(e)}")
        return None

def normalize_text(text):
    """Normalize text for better matching"""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def find_medical_term(text):
    """Find medical term using improved matching logic"""
    normalized_text = normalize_text(text)
    logger.info(f"🔍 Searching for medical term in: '{normalized_text}'")
    
    # 1. Direct term matching (exact substring match)
    for term in MEDICAL_TERMS.keys():
        if term in normalized_text:
            logger.info(f"✅ Found direct match: {term}")
            return MEDICAL_TERMS[term]
    
    # 2. Synonym matching  
    for synonym, medical_term in SYNONYMS.items():
        if synonym in normalized_text:
            logger.info(f"✅ Found synonym match: {synonym} -> {medical_term}")
            if medical_term in MEDICAL_TERMS:
                return MEDICAL_TERMS[medical_term]
    
    # 3. Word-based matching (split into words and check)
    words = normalized_text.split()
    for term in MEDICAL_TERMS.keys():
        term_words = term.split()
        # Check if all words of the term are present in the input
        if all(word in words for word in term_words):
            logger.info(f"✅ Found word-based match: {term}")
            return MEDICAL_TERMS[term]
    
    # 4. Partial matching for compound terms
    for term in MEDICAL_TERMS.keys():
        term_words = set(term.split())
        input_words = set(words)
        # If more than half of the term words are in the input
        if len(term_words.intersection(input_words)) >= max(1, len(term_words) * 0.6):
            logger.info(f"✅ Found partial match: {term}")
            return MEDICAL_TERMS[term]
    
    logger.info("❌ No medical term found")
    return None

def fallback_simplification(text):
    """Improved fallback simplification with better matching"""
    logger.info(f"🔍 FALLBACK: Processing '{text}'")
    
    found_term = find_medical_term(text)
    
    if found_term:
        logger.info("✅ Found medical term in dictionary")
        return found_term
    
    logger.info("❌ No medical term found, returning default message")
    # If no match found, return default message
    return {
        "simple": "I don't have information about this specific condition yet. Please consult with a healthcare professional for accurate medical information.",
        "symptoms": "Unknown - please consult a doctor",
        "causes": "Unknown - please consult a doctor", 
        "prevention": "General advice: maintain a healthy lifestyle with good diet and exercise"
    }

def simplify_medical_text(text):
    logger.info(f"🎯 MAIN: Processing input '{text}'")
    
    # Force fallback mode for testing - comment out the AI check temporarily
    ai_result = None  # Force to use fallback
    # ai_result = simplify_with_ai_model(text)
    
    if ai_result and len(ai_result.strip()) > 10:
        logger.info("🤖 Using AI result")
        return {
            "simple": ai_result,
            "symptoms": "For specific symptoms, please consult a healthcare professional",
            "causes": "For detailed causes, please consult a healthcare professional",
            "prevention": "For prevention strategies, please consult a healthcare professional",
            "source": "AI-generated explanation"
        }
    else:
        logger.info("📚 Using fallback dictionary method")
        result = fallback_simplification(text)
        result["source"] = "Dictionary-based explanation"
        return result

@app.route('/', methods=['GET', 'POST'])
def index():
    simplified = None
    condition = None

    if request.method == 'POST':
        condition = request.form.get('condition', '').strip()
        logger.info(f"📝 Form received condition: '{condition}'")
        if condition:
            simplified = simplify_medical_text(condition)
            logger.info(f"📋 Result: {simplified}")

    return render_template('index.html', simplified=simplified, condition=condition)

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    medical_text = data.get('text', '').strip()
    difficulty = data.get('difficulty', 'simple')
    
    logger.info(f"📝 API received text: '{medical_text}'")
    
    if not medical_text:
        return jsonify({'error': 'Please enter a medical condition to explain'}), 400
    
    explanation = simplify_medical_text(medical_text)
    
    response = {
        'original': medical_text,
        'explanation': explanation,
        'difficulty': difficulty,
        'model_status': 'loaded' if simplifier_pipeline else 'fallback'
    }
    
    return jsonify(response)

@app.route('/model-status')
def model_status():
    return jsonify({
        'model_loaded': simplifier_pipeline is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/reload-model', methods=['POST'])
def reload_model():
    try:
        initialize_model()
        return jsonify({'status': 'success', 'message': 'Model reloaded successfully.'})
    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to reload model: {str(e)}'}), 500

# Debug route to test matching
@app.route('/test-match/<path:text>')
def test_match(text):
    """Debug route to test the matching logic"""
    result = find_medical_term(text)
    return jsonify({
        'input': text,
        'normalized': normalize_text(text),
        'found': result is not None,
        'result': result
    })

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, port=5000)