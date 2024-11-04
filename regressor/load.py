

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer from the fine-tuned model path
model_path = './my_awesome_model2/checkpoint-31500'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.to(device)
model.eval()

def make_prediction(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
    
    # Extract positive and negative sentiment scores
    pos_score = predictions[0][0].item()
    neg_score = predictions[0][1].item()
    return {'positive': pos_score, 'negative': neg_score}

    
with open('validate_data.json', 'r') as f:
    data = json.load(f)
    
for item in data:
    text = item['text']
    prediction = make_prediction(text)
    item['prediction'] = prediction

correct = 0
total = len(data)

for item in data:
    text = item['text']
    prediction = make_prediction(text)

    # Compare prediction with the first label value (assuming positive sentiment is indicated by higher values)
    predicted_label = 'positive' if prediction['positive'] > prediction['negative'] else 'negative'
    actual_label = 'positive' if item['labels'][0] > item['labels'][1] else 'negative'

    if predicted_label == actual_label:
        correct += 1

accuracy = correct / total
print(f'Accuracy: {accuracy:.2f}')

# Interactive loop for predictions
while True:
    input_text = input("Enter a sentence: ")
    if input_text == "exit":
        break
    prediction = make_prediction(input_text)
    print(prediction)
    