import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import multiprocessing

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Check if ROCm GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
myDataset = pd.read_json("results(4).json")
myDataset = myDataset[myDataset['sentence'] != '']
myDataset = myDataset.reset_index(drop=True)
print(myDataset.head())
print(myDataset.shape)

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_data = myDataset.apply(tokenize_function, axis=1)

def make_prediction(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)

    # Extract positive and negative sentiment scores
    pos_score = predictions[0][1].item()
    neg_score = predictions[0][0].item()

    return {'positive': pos_score, 'negative': neg_score}

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(labels)


if __name__ == '__main__':

    #dataset = SentimentDataset(myDataset)

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        myDataset['sentence'].tolist(), 
        myDataset[['pos', 'neg']].values.tolist(), 
        test_size=0.2
    )


    # Create evaluation dataset
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    eval_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=(device.type == 'cpu'))
    val_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, pin_memory=(device.type == 'cpu'))



    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  # Reduced epochs to speed up training
        per_device_train_batch_size=32,  # Increased batch size to speed up training
        per_device_eval_batch_size=32,  # Increased batch size to speed up evaluation
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=1500,  # Save checkpoints more frequently
        logging_steps=500,  # Log training metrics every 500 steps
        eval_strategy='steps',  # Evaluate every 500 steps
        fp16=True,  # Use mixed precision training for faster training
        dataloader_num_workers=6,  # Utilize all CPU cores for data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    trainer.evaluate()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    # Load the fine-tuned model and tokenizer
    model_path = './fine_tuned_model'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

    # Loop to get user input and make predictions
    loop = True
    while loop:
        input_text = input("Please enter your input text: ")
        if input_text.lower() == "exit":
            loop = False
            break
        prediction = make_prediction(input_text)
        print(prediction)