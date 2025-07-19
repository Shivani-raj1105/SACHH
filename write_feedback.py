import csv

feedback = {
    "text": "Example feedback text",
    "model_prediction": "Fake",
    "confidence": 0.95,
    "explanation": "The text contains known fake news patterns.",
    "feedback_type": "user"
}

with open('feedback.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["text", "model_prediction", "confidence", "explanation", "feedback_type"])
    writer.writerow(feedback)