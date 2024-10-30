import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertPreTrainedModel, DataCollatorWithPadding
import torch.nn as nn
import json


LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the data
with open('results_new.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare data
texts = df['sentence'].tolist()
pos = df['pos'].tolist()
neg = df['neg'].tolist()

# Split data into training, validation, and test sets
train_texts, temp_texts, train_pos, temp_pos, train_neg, temp_neg = train_test_split(texts, pos, neg, test_size=0.3, random_state=42)
val_texts, test_texts, val_pos, test_pos, val_neg, test_neg = train_test_split(temp_texts, temp_pos, temp_neg, test_size=0.5, random_state=42)

# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, pos, neg, tokenizer, max_len):
        self.texts = texts
        self.pos = pos
        self.neg = neg
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        pos = self.pos[idx]
        neg = self.neg[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pos': torch.tensor(pos, dtype=torch.float),
            'neg': torch.tensor(neg, dtype=torch.float)
        }

# Create datasets
train_dataset = SentimentDataset(train_texts, train_pos, train_neg, tokenizer, max_len=MAX_LENGTH)
val_dataset = SentimentDataset(val_texts, val_pos, val_neg, tokenizer, max_len=MAX_LENGTH)
test_dataset = SentimentDataset(test_texts, test_pos, test_neg, tokenizer, max_len=MAX_LENGTH)


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define the model
class BertForSentimentRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSentimentRegression, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.3)
        self.linear_pos = nn.Linear(config.hidden_size, 1)
        self.linear_neg = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pos = self.linear_pos(pooled_output)
        neg = self.linear_neg(pooled_output)
        return pos, neg

model = BertForSentimentRegression.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results4',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs4',
    logging_steps=100,
    eval_strategy='steps',
    eval_steps=4000,
    save_steps=4000,
    save_total_limit=4,
    fp16=True,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    max_grad_norm=10.0,
    load_best_model_at_end=True,
    label_names=['pos', 'neg']
)

# Custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # Ensure 'pos' and 'neg' keys are in inputs
        if 'pos' not in inputs or 'neg' not in inputs:
            raise KeyError("The inputs dictionary does not contain 'pos' or 'neg' keys.")
        
        labels_pos = inputs.pop("pos")
        labels_neg = inputs.pop("neg")
        outputs = model(**inputs)
        pos, neg = outputs
        loss_fct = torch.nn.MSELoss()
        loss_pos = loss_fct(pos.view(-1), labels_pos.view(-1))
        loss_neg = loss_fct(neg.view(-1), labels_neg.view(-1))
        loss = loss_pos + loss_neg
        loss = loss.mean()  # Ensure the loss is a scalar
        return (loss, outputs) if return_outputs else loss

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model("./saved_model")

# Evaluate on the test dataset
test_results = trainer.evaluate(test_dataset)
print(test_results)

# Inference function
def predict(text):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    model.eval()
    with torch.no_grad():
        pos, neg = model(input_ids=input_ids, attention_mask=attention_mask)
    return {'pos': pos.item(), 'neg': neg.item()}

# Example usage
print(predict("I like apples"))
print(predict("I hate apples"))