import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
myDataset = pd.read_json("results_new.json")

# Ensure the DataFrame contains 'sentence', 'pos', and 'neg' columns
required_columns = ['sentence', 'pos', 'neg']
if not all(column in myDataset.columns for column in required_columns):
    # If columns are named differently, rename them accordingly
    myDataset.rename(columns={'text': 'sentence', 'positive': 'pos', 'negative': 'neg'}, inplace=True)

# Remove entries with empty sentences
myDataset = myDataset[myDataset['sentence'].astype(bool)]
myDataset = myDataset.reset_index(drop=True)

# Create the final DataFrame
new_df = myDataset[['sentence', 'pos', 'neg']].copy()
print(new_df.head())

# Split the data into training and testing sets
train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)
print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#Data Loaded


#Data Preprocessing
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['sentence'], truncation=True)
    # Use 'pos' and 'neg' as labels
    inputs['labels'] = np.stack((examples['pos'], examples['neg']), axis=1)
    return inputs


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#Load Evaluation function
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Compute metrics for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Ensure predictions and labels are numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    # Separate 'pos' and 'neg' predictions and labels
    preds_pos = predictions[:, 0]
    preds_neg = predictions[:, 1]
    labels_pos = labels[:, 0]
    labels_neg = labels[:, 1]
    # Compute MSE and MAE for 'pos'
    mse_pos = mean_squared_error(labels_pos, preds_pos)
    mae_pos = mean_absolute_error(labels_pos, preds_pos)
    # Compute MSE and MAE for 'neg'
    mse_neg = mean_squared_error(labels_neg, preds_neg)
    mae_neg = mean_absolute_error(labels_neg, preds_neg)
    return {
        'mse_pos': mse_pos,
        'mse_neg': mse_neg,
        'mae_pos': mae_pos,
        'mae_neg': mae_neg
    }

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel

class CustomDistilBertForRegression(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, 2)

        # Apply ReLU for pos_score and negative ReLU for neg_score
        pos_score = self.relu(logits[:, 0])
        neg_score = -self.relu(-logits[:, 1])
        logits = torch.stack((pos_score, neg_score), dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


# Modify the model for regression
model = CustomDistilBertForRegression.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    problem_type='regression'
)

# Sections of config
# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 6
EPOCHS = 5
LEARNING_RATE = 2e-5


# TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='results4',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs4',
    logging_steps=100,
    eval_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    save_total_limit=4,
    fp16=True,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    max_grad_norm=10.0,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# # Save the model
model.save_pretrained('./model4/fine_tuned_model')
tokenizer.save_pretrained('./model4/tokenizer')


