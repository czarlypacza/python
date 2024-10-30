import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch import cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
myDataset = pd.read_json("results(4).json")
myDataset = myDataset[myDataset['sentence'] != '']
myDataset = myDataset.reset_index(drop=True)

# Ensure the DataFrame contains 'sentence', 'pos', and 'neg' columns
new_df = myDataset[['sentence', 'pos', 'neg']].copy()
print(new_df.head())


# Sections of config
# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe['sentence']
        self.targets = dataframe[['pos', 'neg']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        inputs = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids = inputs['input_ids'].flatten()
        mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.flatten()
        targets = torch.tensor(self.targets[index], dtype=torch.float)
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'token_type_ids': token_type_ids,
            'labels': targets
        }

train_size = 0.8
train_dataset = new_df.sample(frac=train_size, random_state=200)
test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = SentimentDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = SentimentDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Model Definition
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)



# TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_steps=1000,
    save_total_limit=4,
    fp16=True,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    max_grad_norm=10.0,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
    tokenizer=tokenizer
)

# Training
trainer.train()

# Evaluation
trainer.evaluate()


# # Save the model
model.save_pretrained('./model2/fine_tuned_model')
tokenizer.save_pretrained('./model2/tokenizer')


def make_prediction(input_text):
    encoding = tokenizer(input_text, return_tensors="pt")
    encoding = {k: v.to(device) for k,v in encoding.items()}

    # Perform inference
    with torch.no_grad():
        outputs = trainer.model(**encoding)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)

    # Extract scores
    pos_score = predictions[0][1].item()
    neg_score = predictions[0][0].item()
    return {'positive': pos_score, 'negative': neg_score}

loop = True
while loop:
    input_text = input("Please enter your input text: ")
    if input_text.lower() == "exit":
        loop = False
        break
    prediction = make_prediction(input_text)
    print(prediction)