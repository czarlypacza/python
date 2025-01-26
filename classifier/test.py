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
    start_time = time.time()
    analysis_results = [make_prediction(item["text"]) for item in test_data]
    time_taken = time.time() - start_time

    correct = 0
    total = len(test_data)
    
    # Positive metrics
    true_positives = 0
    false_negatives_positive = 0
    false_positives = 0

    # Negative metrics
    true_negatives = 0
    false_negatives_negative = 0
    false_negatives = 0

    true_neutrals = 0
    false_negatives_neutral = 0
    false_neutrals = 0

    neutral_predictions = 0
    neutral_positive_misses = 0
    neutral_negative_misses = 0

    # Determine if the dataset includes the neutral class
    neutral_in_dataset = any(item['sentiment'] == 'neutral' for item in test_data)

    for i in range(len(test_data)):
        expected = test_data[i]["sentiment"]
        predicted, _ = analysis_results[i]
        
        
        if predicted == 'neutral':
            neutral_predictions += 1
            if expected == 'positive':
                neutral_positive_misses += 1
                false_neutrals += 1
                false_negatives_positive += 1
            elif expected == 'negative':
                neutral_negative_misses += 1
                false_neutrals += 1
                false_negatives_negative += 1
            elif expected == 'neutral':
                correct += 1
                true_neutrals += 1
        elif predicted == 'positive':
            if expected == 'positive':
                correct += 1
                true_positives += 1
            elif expected == 'negative':
                false_positives += 1
                false_negatives_negative += 1
            elif expected == 'neutral':
                false_negatives_neutral += 1
                false_positives += 1
        elif predicted == 'negative':
            if expected == 'negative':
                correct += 1
                true_negatives += 1
            elif expected == 'positive':
                false_negatives += 1
                false_negatives_positive += 1
            elif expected == 'neutral':
                false_negatives += 1
                false_negatives_neutral += 1

        
    accuracy = correct / total if total else 0

    # Calculate base metrics
    precision_positive = true_positives / (true_positives + false_positives or 1)
    recall_positive = true_positives / (true_positives + false_negatives_positive or 1)
    f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive or 1)

    precision_negative = true_negatives / (true_negatives + false_negatives or 1)
    recall_negative = true_negatives / (true_negatives + false_negatives_negative or 1)
    f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative or 1)

    # Calculate neutral metrics if present in dataset
    precision_neutral = None
    recall_neutral = None
    f1_neutral = None

    if neutral_in_dataset:
        precision_neutral = true_neutrals / (true_neutrals + false_neutrals or 1)
        recall_neutral = true_neutrals / (true_neutrals + false_negatives_neutral or 1)
        f1_neutral = 2 * (precision_neutral * recall_neutral) / (precision_neutral + recall_neutral or 1)

    result = {
        'datasetType': dataset_type,
        'timeTaken': int(time_taken * 1000),
        'accuracy': accuracy,
        'precision': {
            'positive': precision_positive,
            'negative': precision_negative
        },
        'recall': {
            'positive': recall_positive,
            'negative': recall_negative
        },
        'f1Score': {
            'positive': f1_positive,
            'negative': f1_negative
        },
        'totalSamples': total,
        'correctPredictions': correct,
        'neutralStats': {
            'total': neutral_predictions,
            'missedPositives': neutral_positive_misses,
            'missedNegatives': neutral_negative_misses,
            'percentage': (neutral_predictions / total) * 100 if total else 0,
            'trueNeutrals': true_neutrals if neutral_in_dataset else None
        }
    }

    # Add neutral metrics if applicable
    if neutral_in_dataset:
        result['precision']['neutral'] = precision_neutral
        result['recall']['neutral'] = recall_neutral
        result['f1Score']['neutral'] = f1_neutral

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
    for i in range(1,4):
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
        with open('./results_BERT_class_'+str(i)+'.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Analysis complete. Results saved to results.json")
        time.sleep(60*5)

if __name__ == '__main__':
    main()