# classification_model.py
# /media/michal/dev1/sentiment/python/myenv/bin/tensorboard --logdir logs


import os
import json
import numpy as np
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


# Load the original data
with open('/media/michal/dev1/sentiment/python/regressor/results_new.json', 'r') as f:
    data = json.load(f)
    
def labelFromScores(pos, neg):
    if pos > neg * -1:
        return 2
    elif pos < neg * -1:
        return 0
    else:
        return 1

# Transform the data and remove records with empty text
transformed_data = []
for entry in data:
    if entry['sentence'].strip():
        transformed_entry = {
            'label': labelFromScores(entry['pos'], entry['neg']),
            'text': entry['sentence']
        }
        transformed_data.append(transformed_entry)


# Split the data into train and test sets using sklearn
train_data, test_data = train_test_split(transformed_data, test_size=0.15, random_state=42)
train_data, validate_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Save the transformed data to JSON files
with open('./data/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)
with open('./data/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)
with open('./data/validate_data.json', 'w') as f:
    json.dump(validate_data, f, indent=4)



# Define labels
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k.upper() for k, v in label2id.items()}


def load_data(data_dir):
    data_files = {
        "train": str(data_dir / "train_data.json"),
        "validation": str(data_dir / "validate_data.json"),
        "test": str(data_dir / "test_data.json"),
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    data_dir = Path("./data")
    dataset = load_data(data_dir)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./sentiment_classifier",
        num_train_epochs=3,
        per_device_train_batch_size=48,  # Increase batch size to utilize more GPU memory
        per_device_eval_batch_size=48,   # Increase batch size to utilize more GPU memory
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        save_steps=2000,
        eval_steps=2000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(f"Evaluation results: {eval_results}")

    # Save the model
    trainer.save_model("./sentiment_classifier_model")

    
    # Example usage after training
    import torch

    def get_sentiment_scores(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()
            # Map probabilities to labels
            positive_score = probabilities[label2id['positive']]
            neutral_score = probabilities[label2id['neutral']]
            negative_score = probabilities[label2id['negative']]
        return {
            "positive_score": positive_score,
            "neutral_score": neutral_score,
            "negative_score": negative_score
        }

    # Example usage
    text = "I like this company. They were very helpful and friendly."
    scores = get_sentiment_scores(text)
    print(f"Text: {text}")
    print(f"Scores: {scores}")

if __name__ == "__main__":
    main()
    