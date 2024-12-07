# test.py

import time
import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./sentiment_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_classifier_model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Label mappings
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

def make_prediction(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()
    
    # Get predicted label and probabilities
    predicted_label_id = probabilities.argmax()
    predicted_label = id2label[predicted_label_id]
    return predicted_label, probabilities

def calculate_metrics(test_data, dataset_type):
    total = len(test_data)
    correct = 0
    # Initialize counts
    true_positives = true_negatives = true_neutrals = 0
    false_positives = false_negatives = false_neutrals = 0
    neutral_predictions = 0
    neutral_positive_misses = neutral_negative_misses = 0

    # Determine if the dataset includes the neutral class
    neutral_in_dataset = any(item['sentiment'] == 'neutral' for item in test_data)

    # Start timing only predictions
    prediction_time = 0.0
    for item in test_data:
        start_time = time.time()
        predicted_label, probabilities = make_prediction(item['text'])
        prediction_time += time.time() - start_time

        true_label = item['sentiment']

        if predicted_label == true_label:
            correct += 1
            if predicted_label == 'positive':
                true_positives += 1
            elif predicted_label == 'negative':
                true_negatives += 1
            elif predicted_label == 'neutral':
                true_neutrals += 1
        else:
            if predicted_label == 'positive':
                if true_label == 'negative' or true_label == 'neutral':
                    false_positives += 1
            elif predicted_label == 'negative':
                if true_label == 'positive' or true_label == 'neutral':
                    false_negatives += 1
            elif predicted_label == 'neutral':
                false_neutrals += 1
                if true_label == 'positive':
                    neutral_positive_misses += 1
                elif true_label == 'negative':
                    neutral_negative_misses += 1

        # For neutralStats
        if predicted_label == 'neutral':
            neutral_predictions += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    precision_pos = true_positives / (true_positives + false_positives or 1)
    recall_pos = true_positives / (true_positives + false_negatives or 1)
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos or 1)

    precision_neg = true_negatives / (true_negatives + false_negatives or 1)
    recall_neg = true_negatives / (true_negatives + false_positives or 1)
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg or 1)

    # Calculate neutral metrics if neutral class is present
    if neutral_in_dataset:
        precision_neu = true_neutrals / (true_neutrals + false_neutrals or 1)
        recall_neu = true_neutrals / (true_neutrals + false_neutrals or 1)
        f1_neu = 2 * (precision_neu * recall_neu) / (precision_neu + recall_neu or 1)
    else:
        precision_neu = recall_neu = f1_neu = None

    result = {
        "datasetType": dataset_type,
        "timeTaken": int(prediction_time * 1000),  # Convert to milliseconds
        "accuracy": accuracy,
        "precision": {
            "positive": precision_pos,
            "negative": precision_neg
        },
        "recall": {
            "positive": recall_pos,
            "negative": recall_neg
        },
        "f1Score": {
            "positive": f1_pos,
            "negative": f1_neg
        },
        "totalSamples": total,
        "correctPredictions": correct,
        "neutralStats": {
            "total": neutral_predictions,
            "missedPositives": neutral_positive_misses,
            "missedNegatives": neutral_negative_misses,
            "percentage": (neutral_predictions / total) * 100,
            "trueNeutrals": true_neutrals if neutral_in_dataset else None
        }
    }

    # Add neutral metrics if applicable
    if neutral_in_dataset:
        result["precision"]["neutral"] = precision_neu
        result["recall"]["neutral"] = recall_neu
        result["f1Score"]["neutral"] = f1_neu

    return result

def analyze_standard_dataset(data_dir):
    files = ['imdb_labelled.txt', 'yelp_labelled.txt', 'amazon_cells_labelled.txt']
    test_data = []
    
    for file in files:
        file_path = data_dir / file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sentence, score = line.strip().split('\t')
                        sentiment = 'positive' if int(score) == 1 else 'negative'
                        test_data.append({"text": sentence, "sentiment": sentiment})
                    except ValueError:
                        continue  # Skip lines that don't have the correct format
    return calculate_metrics(test_data, "standard")

def analyze_twitter_dataset(data_dir):
    twitter_file = data_dir / 'test_twitter.csv'
    twitter_data = []
    
    with open(twitter_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # CSV format: "sentiment","id","date","query","user","text"
                parts = line.strip().split('","')
                if len(parts) >= 6:
                    sentiment = parts[0].replace('"', '')
                    text = parts[5].replace('"', '')
                    
                    # Map sentiment values
                    sentiment_map = {
                        '4': 'positive',
                        '2': 'neutral',
                        '0': 'negative'
                    }
                    mapped_sentiment = sentiment_map.get(sentiment.rstrip(','))
                    
                    if mapped_sentiment:
                        twitter_data.append({"text": text, "sentiment": mapped_sentiment})
    return calculate_metrics(twitter_data, "twitter")

def analyze_additional_dataset(data_dir):
    additional_file = data_dir / 'train.jsonl'
    additional_data = []

    with open(additional_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                additional_data.append({"text": entry["text"], "sentiment": entry["label_text"]})
    return calculate_metrics(additional_data, "additional")

def main():
    data_dir = Path('/media/michal/dev1/sentiment/python-sentiment/data')
    
    # Analyze standard dataset
    standard_results = analyze_standard_dataset(data_dir)
    
    # Analyze Twitter dataset
    twitter_results = analyze_twitter_dataset(data_dir)
    
    # Analyze additional dataset
    additional_results = analyze_additional_dataset(data_dir)
    
    # Save results
    results = {
        'standardResults': standard_results,
        'twitterResults': twitter_results,
        'additionalResults': additional_results
    }
    with open('./results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete. Results saved to results.json")

if __name__ == '__main__':
    main()