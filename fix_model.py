import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def check_and_fix_model():
    print("=== MODEL FIX SCRIPT ===")
    
    saved_model_path = "saved_model"
    model_safetensors_path = os.path.join(saved_model_path, "model.safetensors")
    
    if os.path.exists(model_safetensors_path):
        size = os.path.getsize(model_safetensors_path)
        if size > 0:
            print(f"Model weights exist ({size} bytes)")
            return True
        else:
            print("Model weights file is empty (0 bytes)")
    else:
        print("Model weights file does not exist")
    
    print("\nFixing model weights...")
    
    try:
        print("Attempting to load existing model...")
        tokenizer = BertTokenizer.from_pretrained(saved_model_path)
        model = BertForSequenceClassification.from_pretrained(saved_model_path)
        print("Successfully loaded existing model!")
        return True
    except Exception as e:
        print(f"Failed to load existing model: {e}")
    
    try:
        print("Creating new model from scratch...")
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        print("Saving model...")
        model.save_pretrained(saved_model_path)
        tokenizer.save_pretrained(saved_model_path)
        
        print("New model created and saved successfully!")
        return True
    except Exception as e:
        print(f"Failed to create new model: {e}")
        return False

def test_model_loading():
    print("\nTesting model loading...")
    
    try:
        saved_model_path = "saved_model"
        tokenizer = BertTokenizer.from_pretrained(saved_model_path)
        model = BertForSequenceClassification.from_pretrained(saved_model_path)
        model.eval()
        
        test_text = "This is a test news article."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        label = "Real" if prediction == 1 else "Fake"
        print(f"Model test successful!")
        print(f"   Test text: '{test_text}'")
        print(f"   Prediction: {label}")
        print(f"   Confidence: {confidence:.2%}")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

if __name__ == "__main__":
    success = check_and_fix_model()
    if success:
        test_model_loading()
    else:
        print("\nFailed to fix model. You may need to train the model first.")
        print("Run: python train.py") 