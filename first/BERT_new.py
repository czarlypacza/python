import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel,BertPreTrainedModel, Trainer, TrainingArguments, BertConfig 

from torch import cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
myDataset = pd.read_json("results(4) copy.json")
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
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe['sentence']  # Assuming your column name is 'sentence'
        self.targets = dataframe[['pos', 'neg']].values  # Ensure this returns a 2D array
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())  # Clean extra spaces

        inputs = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # Updated to current usage
            truncation=True,  # Enforces truncation if the tokenized text exceeds MAX_LEN
            return_tensors='pt'  # Return PyTorch tensors directly
        )

        # Extracting the input IDs, attention masks, and token type IDs
        ids = inputs['input_ids'].flatten()  # Flattening to ensure correct shape
        mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.flatten()  # Flatten if it's not None

        # Prepare targets as a tensor, ensuring correct dtype
        targets = torch.tensor(self.targets[index], dtype=torch.float16)

        # Return the dictionary of inputs and targets
        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets': targets
        }

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = SentimentDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = SentimentDataset(test_dataset, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss(reduction='mean')(outputs, targets)  

# Model Definition
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids=None, targets=None):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        logits = self.l3(output_2)

        if targets is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fn(logits, targets)
            return {"loss": loss, "logits": logits}  # Return a dictionary with loss and logits
        
        return {"logits": logits}  # Return a dictionary with logits for inference/evaluation
    
    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)
        
    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
    
  

model = BERTClass()
model.to(device)

# TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='steps',
    eval_steps=1000,
    save_steps=1000,
    fp16=True,
    learning_rate=LEARNING_RATE,  # Ensure the learning rate is set
    gradient_accumulation_steps=1,  # Adjust as needed
    max_grad_norm=1.0,  # Gradient clipping
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
)



# Training
trainer.train()

# Evaluation
trainer.evaluate()


#optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# def train_model(epoch):
#     model.train()
#     for _, data in enumerate(training_loader, 0):
#         ids = data['ids'].to(device, dtype=torch.long)
#         mask = data['mask'].to(device, dtype=torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
#         targets = data['targets'].to(device, dtype=torch.float)

#         outputs = model(ids, mask, token_type_ids)
#         loss = loss_fn(outputs, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if _ % 100 == 0:
#             print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
# # Evaluation Function
# def validation(epoch):
#     model.eval()
#     fin_targets=[]
#     fin_outputs=[]
#     with torch.no_grad():
#         for _, data in enumerate(testing_loader, 0):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
#             outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
#     return fin_outputs, fin_targets


# # Training loop
# for epoch in range(EPOCHS):
#     train_model(epoch)


# for epoch in range(EPOCHS):
#     outputs, targets = validation(epoch)
#     outputs = np.array(outputs) >= 0.5
#     accuracy = metrics.accuracy_score(targets, outputs)
#     f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
#     f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
#     print(f"Accuracy Score = {accuracy}")
#     print(f"F1 Score (Micro) = {f1_score_micro}")
#     print(f"F1 Score (Macro) = {f1_score_macro}")

# # Save the model
model.save_pretrained('./model1/fine_tuned_model')
tokenizer.save_pretrained('./model1/tokenizer')


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

# Example usage
input_text = "I hate this company"
prediction = make_prediction(input_text)
print(prediction)