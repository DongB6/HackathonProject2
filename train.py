from load_dataset import load_and_preprocess
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

def setup_training():
    """Setup directories and check CUDA availability"""
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/med_simplifier", exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def load_and_split_data():
    """Load data and split into train/validation sets"""
    print("Loading and preprocessing data...")
    data = load_and_preprocess()
    
    if not data:
        raise ValueError("No data loaded! Check your preprocessing script.")
    
    print(f"Total dataset size: {len(data)}")
    
    # Split data into train/validation (80/20)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Preview examples
    print("\n" + "="*50)
    print("SAMPLE TRAINING EXAMPLES:")
    print("="*50)
    for i in range(min(3, len(train_data))):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {train_data[i]['instruction']}")
        print(f"Input: {train_data[i]['input'][:100]}..." if len(train_data[i]['input']) > 100 else f"Input: {train_data[i]['input']}")
        print(f"Output: {train_data[i]['output'][:100]}..." if len(train_data[i]['output']) > 100 else f"Output: {train_data[i]['output']}")
    
    return train_data, val_data

def tokenize_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tokenize examples for T5 model"""
    # Combine instruction and input for T5
    inputs = [f"{inst} {inp}" for inst, inp in zip(examples['instruction'], examples['input'])]
    targets = examples['output']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding=False 
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding=False
    )
    
    # T5 expects labels, not target_ids
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def compute_metrics(eval_pred):
    """Compute metrics during evaluation"""
    predictions, labels = eval_pred
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple metrics - you can add BLEU, ROUGE etc.
    avg_pred_length = np.mean([len(pred.split()) for pred in decoded_preds])
    avg_label_length = np.mean([len(label.split()) for label in decoded_labels])
    
    return {
        "avg_pred_length": avg_pred_length,
        "avg_label_length": avg_label_length,
    }

def main():
    """Main training pipeline"""
    print("🚀 Starting Medical Text Simplification Training")
    print("="*60)
    
    # Setup
    device = setup_training()
    
    # Load and split data
    train_data, val_data = load_and_split_data()
    
    # Initialize model and tokenizer
    print("\n📥 Loading model and tokenizer...")
    model_name = "google/flan-t5-small"  # You can also try "flan-t5-base" for better results
    
    global tokenizer  # Make tokenizer global for compute_metrics
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = {"additional_special_tokens": ["<medical>", "<simple>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Create datasets
    print("\n🔄 Creating datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    
    # Training arguments
    print("\n⚙️ Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./models/med_simplifier",
        
        # Training hyperparameters
        num_train_epochs=3,  # Start with fewer epochs
        per_device_train_batch_size=4 if torch.cuda.is_available() else 2,
        per_device_eval_batch_size=4 if torch.cuda.is_available() else 2,
        gradient_accumulation_steps=2,  # Effective batch size = batch_size * gradient_accumulation_steps
        
        # Optimization
        learning_rate=5e-4,  # Slightly higher learning rate for fine-tuning
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Mixed precision
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        
        # Logging and saving
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        
        # Evaluation
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Other
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        seed=42,
    )
    
    # Create trainer
    print("🏋️ Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("\n🔥 Starting training...")
    print("="*60)
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        
        # Save final model
        print("💾 Saving model and tokenizer...")
        trainer.save_model("./models/med_simplifier")
        tokenizer.save_pretrained("./models/med_simplifier")
        
        # Save training info
        training_info = {
            "model_name": model_name,
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "epochs": training_args.num_train_epochs,
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", "N/A"),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
        }
        
        with open("./models/med_simplifier/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"📊 Training Summary:")
        print(f"   Model: {model_name}")
        print(f"   Training examples: {len(train_data):,}")
        print(f"   Validation examples: {len(val_data):,}")
        print(f"   Epochs: {training_args.num_train_epochs}")
        print(f"   Model saved to: ./models/med_simplifier")
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise
    
    # Test the model
    print("\n🧪 Testing the trained model...")
    test_model()

def test_model():
    """Test the trained model with sample inputs"""
    try:
        from transformers import pipeline
        
        # Load the trained model
        simplifier = pipeline(
            "text2text-generation",
            model="./models/med_simplifier",
            tokenizer="./models/med_simplifier",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Test cases
        test_cases = [
            "Explain this medical term in simple language: myocardial infarction",
            "Simplify this medical information: hypertension",
            "What is diabetes mellitus?",
        ]
        
        print("Sample outputs:")
        for i, test_case in enumerate(test_cases):
            result = simplifier(test_case, max_length=100, clean_up_tokenization_spaces=True)
            print(f"{i+1}. Input: {test_case}")
            print(f"   Output: {result[0]['generated_text']}")
            print()
            
    except Exception as e:
        print(f"Testing failed: {str(e)}")

if __name__ == "__main__":
    main()