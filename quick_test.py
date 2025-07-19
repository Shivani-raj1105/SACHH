
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def quick_test():
    st.title("Quick Model Test")
    
    # Test text
    test_text = st.text_area("Enter text to test:", 
                             "Scientists discover new species of dinosaur in Argentina")
    
    if st.button("Test Original Model"):
        try:
            tokenizer = AutoTokenizer.from_pretrained("saved_model")
            model = AutoModelForSequenceClassification.from_pretrained("saved_model")
            
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            label = "Real" if prediction == 1 else "Fake"
            st.success(f"Original Model: {label} (Confidence: {confidence:.3f})")
        except Exception as e:
            st.error(f"Error testing original model: {e}")
    
    if st.button("Test Improved Model"):
        try:
            tokenizer = AutoTokenizer.from_pretrained("saved_model_improved")
            model = AutoModelForSequenceClassification.from_pretrained("saved_model_improved")
            
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            label = "Real" if prediction == 1 else "Fake"
            st.success(f"Improved Model: {label} (Confidence: {confidence:.3f})")
        except Exception as e:
            st.error(f"Error testing improved model: {e}")

if __name__ == "__main__":
    quick_test()
