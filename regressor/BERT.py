import json
from datasets import DatasetDict, Dataset, load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Load the original data
with open('results_new.json', 'r') as f:
    data = json.load(f)

# Transform the data and remove records with empty text
transformed_data = []
for entry in data:
    if entry['sentence'].strip():
        transformed_entry = {
            'labels': [entry['pos'], -entry['neg']],
            'text': entry['sentence']
        }
        transformed_data.append(transformed_entry)


# Split the data into train and test sets using sklearn
train_data, test_data = train_test_split(transformed_data, test_size=0.15, random_state=42)
train_data, validate_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Save the transformed data to JSON files
with open('train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)
with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)
with open('validate_data.json', 'w') as f:
    json.dump(validate_data, f, indent=4)


# Load the dataset using load_dataset
dataset = load_dataset('json', data_files={
    'train': 'train_data.json',
    'test': 'test_data.json',
    'validate': 'validate_data.json'
})



import matplotlib.pyplot as plt

# # Visualize the distribution of labels in the dataset
# def plot_label_distribution(data, title):
#     labels = [entry['label'] for entry in data]
#     plt.hist(labels, bins=3, edgecolor='black')
#     plt.xticks([0, 1, 2], ['Negative','Neutral', 'Positive'])
#     plt.xlabel('Label')
#     plt.ylabel('Count')
#     plt.title(title)
#     plt.savefig(title + '.png')

# # Load the transformed data
# with open('train_data.json', 'r') as f:
#     train_data = json.load(f)
# with open('test_data.json', 'r') as f:
#     test_data = json.load(f)
# with open('validate_data.json', 'r') as f:
#     validate_data = json.load(f)

# # Plot the label distribution for train and test sets
# plot_label_distribution(train_data, 'Train Data Label Distribution')
# plot_label_distribution(test_data, 'Test Data Label Distribution')
# plot_label_distribution(validate_data, 'Validate Data Label Distribution')


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "mae": mae, "r2": r2}

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}



model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
    problem_type="regression"
).to("cuda")

training_args = TrainingArguments(
    output_dir="./my_awesome_model2",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_steps=500,
    save_total_limit=2,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],    
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


print(trainer.evaluate(eval_dataset=tokenized_dataset["validate"]))

trainer.save_model("./my_awesome_model")

# Test the model on a test inference

sentence = "I like this company. they were very helpful and friendly."
inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
outputs = model(**inputs)
positive_score, negative_score = outputs.logits[0].tolist()
print(f"Positive sentiment score: {positive_score}")
print(f"Negative sentiment score: {negative_score}")
