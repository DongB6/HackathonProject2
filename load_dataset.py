import json
import os
import re

def load_and_preprocess():
    """Load and preprocess medical data for text simplification training"""
    json_path = os.path.join("output", "medical_terms_combined.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run the data collection script first.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("terms", [])
    converted = []
    
    for entry in entries:
        title = entry.get("title", "").strip()
        description = entry.get("description", "").strip()
        symptoms = entry.get("symptoms", "").strip()
        causes = entry.get("causes", "").strip()
        prevention = entry.get("prevention", "").strip()
        treatment = entry.get("treatment", "").strip()
        also_called = entry.get("also_called", [])
        
        # Skip entries without sufficient information
        if not title or not description:
            continue
            
        # Create multiple training examples from each entry
        examples = []
        
        # 1. Term to Simple Explanation
        if description and len(description) > 20:
            examples.append({
                "instruction": "Explain this medical term in simple language that anyone can understand:",
                "input": title,
                "output": clean_and_simplify_text(description)
            })
        
        # 2. Complex description to simple explanation
        if description and len(description) > 50:
            complex_input = f"Medical term: {title}. Definition: {description}"
            simple_output = create_simple_explanation(title, description)
            
            examples.append({
                "instruction": "Simplify this medical information for a patient:",
                "input": complex_input,
                "output": simple_output
            })
        
        # 3. Alternative names
        if also_called:
            alt_names = ", ".join(also_called[:5])  # Limit to #
            examples.append({
                "instruction": "Explain what this medical condition is:",
                "input": f"{title} (also called: {alt_names})",
                "output": clean_and_simplify_text(description)
            })
        
        # 4. Symptoms explanation if available
        if symptoms and len(symptoms) > 10:
            examples.append({
                "instruction": "What are the symptoms of this condition?",
                "input": title,
                "output": clean_and_simplify_text(symptoms)
            })
        
        # 5. Causes explanation if available
        if causes and len(causes) > 10:
            examples.append({
                "instruction": "What causes this medical condition?",
                "input": title,
                "output": clean_and_simplify_text(causes)
            })
        
        # 6. Prevention explanation if available
        if prevention and len(prevention) > 10:
            examples.append({
                "instruction": "How can this condition be prevented?",
                "input": title,
                "output": clean_and_simplify_text(prevention)
            })
        
        converted.extend(examples)
    
    print(f"Generated {len(converted)} training examples from {len(entries)} medical terms")
    return converted

def clean_and_simplify_text(text):
    """Clean and slightly simplify medical text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace some complex terms with simpler ones
    replacements = {
        "myocardial infarction": "heart attack",
        "hypertension": "high blood pressure",
        "diabetes mellitus": "diabetes",
        "cardiovascular": "heart and blood vessel",
        "respiratory": "breathing",
        "gastrointestinal": "stomach and intestine",
        "inflammation": "swelling and irritation",
        "chronic": "long-term",
        "acute": "sudden or severe",
        "benign": "not harmful",
        "malignant": "cancerous",
        "diagnosis": "finding out what's wrong",
        "prognosis": "expected outcome",
        "therapeutic": "treatment-related"
    }
    
    text_lower = text.lower()
    for complex_term, simple_term in replacements.items():
        text_lower = text_lower.replace(complex_term, simple_term)
    
    # Capitalize first letter
    return text_lower[0].upper() + text_lower[1:] if text_lower else ""

def create_simple_explanation(title, description):
    """Create a simple explanation combining title and description"""
    simple_desc = clean_and_simplify_text(description)
    
    # Create a more conversational explanation
    if "heart attack" in title.lower() or "myocardial infarction" in title.lower():
        return f"{title} is a heart attack. This happens when {simple_desc.lower()}"
    elif "diabetes" in title.lower():
        return f"{title} is a condition where {simple_desc.lower()}"
    elif "blood pressure" in title.lower() or "hypertension" in title.lower():
        return f"{title} means {simple_desc.lower()}"
    else:
        return f"{title} is a condition where {simple_desc.lower()}"

def save_training_data(training_data, output_path="training_data.json"):
    """Save preprocessed training data"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data saved to {output_path}")
    
    
    # Format for Hugging Face datasets
    hf_format = []
    for item in training_data:
        hf_format.append({
            "text": f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
        })
    
    hf_path = output_path.replace(".json", "_hf_format.json")
    with open(hf_path, "w", encoding="utf-8") as f:
        json.dump(hf_format, f, indent=2, ensure_ascii=False)
    
    print(f"Hugging Face format saved to {hf_path}")

def main():
    """Main preprocessing pipeline"""
    print("Starting medical data preprocessing...")
    
    # Load and preprocess data
    training_data = load_and_preprocess()
    
    if not training_data:
        print("No training data generated. Check your input files.")
        return
    
    # Save training data
    save_training_data(training_data, "output/training_data.json")
    
    # Print some statistics
    print(f"\nTraining Data Statistics:")
    print(f"Total examples: {len(training_data)}")
    
    # Count instruction types
    instruction_counts = {}
    for item in training_data:
        instruction = item['instruction']
        instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
    
    print(f"\nInstruction distribution:")
    for instruction, count in instruction_counts.items():
        print(f"  {instruction}: {count}")
    
    # Show a few previews examples
    print(f"\nSample training examples:")
    for i, item in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {item['instruction']}")
        print(f"Input: {item['input']}")
        print(f"Output: {item['output']}")

if __name__ == "__main__":
    main()