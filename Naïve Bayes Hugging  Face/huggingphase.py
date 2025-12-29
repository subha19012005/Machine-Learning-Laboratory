from transformers import pipeline

# Load pretrained model
sentiment_model = pipeline("sentiment-analysis")

# Test samples
texts = ["Excellent product", "Worst service ever", "The movie was okay"]

# Predict
results = sentiment_model(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} (Score: {result['score']:.2f})")
    print("-" * 40)
