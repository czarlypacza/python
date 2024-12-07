# interactive_sentiment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_sentiment_scores(text, tokenizer, model, device, label2id):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()
    return probabilities

def main():
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_classifier_model")
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_classifier_model")

    # Move model to device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Label mapping
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {v: k.capitalize() for k, v in label2id.items()}

    print("Sentiment Analysis Model Loaded. Type 'exit' to quit.\n")

    while True:
        text = input("Enter a sentence: ")
        if text.lower() == 'exit':
            break
        if not text.strip():
            print("Please enter a valid sentence.\n")
            continue

        # Get sentiment scores
        probabilities = get_sentiment_scores(text, tokenizer, model, device, label2id)
        predicted_class_id = probabilities.argmax()
        sentiment = id2label[predicted_class_id]

        # Map probabilities to labels
        positive_score = probabilities[label2id['positive']]
        neutral_score = probabilities[label2id['neutral']]
        negative_score = probabilities[label2id['negative']]

        print(f"Predicted sentiment: {sentiment}")
        print(f"Scores -> Positive: {positive_score:.4f}, Neutral: {neutral_score:.4f}, Negative: {negative_score:.4f}\n")

if __name__ == "__main__":
    main()