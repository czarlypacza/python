# regression_model.py

import os
import json
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BertForSentimentRegression(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForSentimentRegression, self).__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        scores = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(scores, labels)

        return {'loss': loss, 'scores': scores}


def load_data(data_dir):
    data_files = {
        "train": str(data_dir / "train_data.json"),
        "validation": str(data_dir / "validate_data.json"),
        "test": str(data_dir / "test_data.json"),
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset

def preprocess_function(examples, tokenizer):
    inputs = tokenizer(examples["text"], truncation=True)
    # Handle array labels directly
    inputs["labels"] = torch.tensor(examples["labels"], dtype=torch.float32)
    return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels.flatten(), predictions.flatten())
    return {"mse": mse, "mae": mae, "r2": r2}


def main():
    data_dir = Path("./")
    dataset = load_data(data_dir)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./new_model",
        num_train_epochs=3,
        per_device_train_batch_size=48,  # Increase batch size to utilize more GPU memory
        per_device_eval_batch_size=48,   # Increase batch size to utilize more GPU memory
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
        save_steps=2000,
        eval_steps=2000,
    )

    def model_init():
        return BertForSentimentRegression(num_labels=2)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(f"Evaluation results: {eval_results}")

    trainer.save_model("./sentiment_regressor_model")


    # Example prediction
    text = "I love this product! It works great."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        scores = trainer.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )['scores']
    pos_score, neg_score = scores[0].tolist()
    print(f"Text: {text}")
    print(f"Positive score: {pos_score}")
    print(f"Negative score: {neg_score}")

if __name__ == "__main__":
    main()