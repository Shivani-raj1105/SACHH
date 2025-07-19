import csv
import os
import time

def test_save_feedback():
    print("Testing feedback saving...")
    print(f"Current directory: {os.getcwd()}")
    print(f"feedback.csv exists: {os.path.exists('feedback.csv')}")
    
    # Test data
    input_text = "zcsFVfrszx"
    label = "Fake"
    confidence = 0.85
    explanation = [("test", 0.5)]
    feedback_type = "not_news"
    
    try:
        # Sanitize input to prevent malformed CSV
        safe_text = input_text.replace('\n', ' ').replace('\r', ' ').replace(',', ' ').strip()
        safe_feedback_type = feedback_type.replace('\n', ' ').replace('\r', ' ').replace(',', ' ').strip()
        feedback_data = [safe_text, label, confidence, str(explanation), safe_feedback_type]
        file_exists = os.path.exists("feedback.csv")
        
        print(f"Feedback data: {feedback_data}")
        print(f"File exists: {file_exists}")
        
        with open("feedback.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                print("Writing header...")
                writer.writerow(["text", "model_prediction", "confidence", "explanation", "feedback_type"])
            print("Writing feedback data...")
            writer.writerow(feedback_data)
        
        print("Feedback saved successfully!")
        
        # Read back to verify
        with open("feedback.csv", "r", encoding="utf-8") as f:
            content = f.read()
            print(f"File content:\n{content}")
            
    except Exception as e:
        print(f"Error saving feedback: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save_feedback() 