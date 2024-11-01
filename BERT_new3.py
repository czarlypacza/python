import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertPreTrainedModel, DataCollatorWithPadding
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

LEARNING_RATE = 5e-5
MAX_LENGTH = 300
BATCH_SIZE = 6
EPOCHS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the data
with open('results_new.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Normalize 'pos' to [0, 1]
df['pos'] = (df['pos'] - df['pos'].min()) / (df['pos'].max() - df['pos'].min())

# Normalize 'neg' to [-1, 0]
neg_max = df['neg'].max()

# Normalize 'neg' to [-1, 0]
neg_min = df['neg'].min()
df['neg'] = (df['neg'] / neg_min) - 1
df['neg'] = -(1+df['neg'])


# Define a threshold to filter out near-zero values
threshold = 0.01

# Filter out rows where both 'pos' and 'neg' are near zero
df_filtered = df[(df['pos'] > threshold) | (df['neg'] < -threshold)]


# Calculate the 99th percentile for 'pos' and the 1st percentile for 'neg'
pos_99th_percentile = df_filtered['pos'].quantile(0.99)
neg_1st_percentile = df_filtered['neg'].quantile(0.02)

# Remove outliers
df_filtered = df_filtered[(df_filtered['pos'] <= pos_99th_percentile) & (df_filtered['neg'] >= neg_1st_percentile)]

print(df_filtered[['pos', 'neg']].head(50))

# Histogram for 'pos' scores
plt.figure(figsize=(8, 6))
plt.hist(df_filtered['pos'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Positive Sentiment Scores')
plt.xlabel('Positive Score')
plt.ylabel('Frequency')
plt.savefig('pos_scores_histogram.png')

# Histogram for 'neg' scores
plt.figure(figsize=(8, 6))
plt.hist(df_filtered['neg'], bins=50, color='red', alpha=0.7)
plt.title('Distribution of Negative Sentiment Scores')
plt.xlabel('Negative Score')
plt.ylabel('Frequency')
plt.savefig('neg_scores_histogram.png')

# Scatter plot for 'pos' vs 'neg' scores
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_filtered, x='pos', y='neg', alpha=0.5)
plt.title('Positive vs Negative Sentiment Scores')
plt.xlabel('Positive Score')
plt.ylabel('Negative Score')
plt.savefig('pos_vs_neg_scatter.png')

# Histogram for text lengths
df_filtered['text_length'] = df_filtered['sentence'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
plt.hist(df_filtered['text_length'], bins=50, color='green', alpha=0.7)
plt.title('Distribution of Text Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig('text_length_histogram.png')

# Word cloud for sentences
text = ' '.join(df_filtered['sentence'])
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', max_words=100).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')

# Box plot for 'pos' and 'neg' scores
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtered[['pos', 'neg']])
plt.title('Box Plot of Sentiment Scores')
plt.ylabel('Score')
plt.savefig('sentiment_scores_boxplot.png')

# Correlation matrix heatmap
corr = df_filtered[['pos', 'neg', 'text_length']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare data
texts = df_filtered['sentence'].tolist()
pos = df_filtered['pos'].tolist()
neg = df_filtered['neg'].tolist()

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
        self.dropout = nn.Dropout(p=0.5)
        self.linear_pos = nn.Linear(config.hidden_size, 1)
        self.linear_neg = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        pos_logits = self.linear_pos(pooled_output)
        neg_logits = self.linear_neg(pooled_output)
        # Apply Sigmoid activation
        pos = torch.sigmoid(pos_logits)
        neg = torch.sigmoid(neg_logits) - 1
        return pos, neg

model = BertForSentimentRegression.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results7',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir='./logs6',
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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_pos = inputs.pop("pos")
        labels_neg = inputs.pop("neg")
        outputs = model(**inputs)
        pos, neg = outputs
        loss_fct = torch.nn.MSELoss()
        loss_pos = loss_fct(pos.view(-1), labels_pos.view(-1))
        loss_neg = loss_fct(neg.view(-1), labels_neg.view(-1))
        loss = loss_pos + loss_neg
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
trainer.save_model("./saved_model3")

# Evaluate on the test dataset
test_results = trainer.evaluate(test_dataset)
print(test_results)

# # Inference function
# def predict(text):
#     encoding = tokenizer(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         return_token_type_ids=False,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#     )
#     input_ids = encoding['input_ids']
#     attention_mask = encoding['attention_mask']
#     model.eval()
#     with torch.no_grad():
#         pos, neg = model(input_ids=input_ids, attention_mask=attention_mask)
#     return {'pos': pos.item(), 'neg': neg.item()}

# # Example usage
# print(predict("I like apples"))
# print(predict("I hate apples"))


# {'eval_loss': 2.8301985263824463, 'eval_runtime': 163.8968, 'eval_samples_per_second': 107.879, 'eval_steps_per_second': 17.981, 'epoch': 6.98}
# {'loss': 2.6348, 'grad_norm': 0.14554549753665924, 'learning_rate': 1.5129543132389431e-05, 'epoch': 6.99}
# {'loss': 2.539, 'grad_norm': 2.1525750160217285, 'learning_rate': 1.509305210918114e-05, 'epoch': 7.0} {'loss': 2.9428, 'grad_norm': 0.8117209672927856, 'learning_rate': 1.505656108597285e-05, 'epoch': 7.0} {'loss': 2.4454, 'grad_norm': 0.7278549671173096, 'learning_rate': 1.502007006276456e-05, 'epoch': 7.01} {'loss': 2.6921, 'grad_norm': 0.6981877684593201, 'learning_rate': 1.498357903955627e-05, 'epoch': 7.02} {'loss': 3.1115, 'grad_norm': 1.573244571685791, 'learning_rate': 1.494708801634798e-05, 'epoch': 7.02} {'loss': 2.5209, 'grad_norm': 1.5326796770095825, 'learning_rate': 1.4910596993139688e-05, 'epoch': 7.03} {'loss': 2.237, 'grad_norm': 0.7953216433525085, 'learning_rate': 1.4874105969931398e-05, 'epoch': 7.04} {'loss': 2.6588, 'grad_norm': 0.2755764126777649, 'learning_rate': 1.4837614946723108e-05, 'epoch': 7.05} {'loss': 2.341, 'grad_norm': 2.672820806503296, 'learning_rate': 1.4801123923514814e-05, 'epoch': 7.05} {'loss': 2.4922, 'grad_norm': 2.1530959606170654, 'learning_rate': 1.4764632900306524e-05, 'epoch': 7.06} {'loss': 2.6206, 'grad_norm': 0.4918759763240814, 'learning_rate': 1.4728141877098234e-05, 'epoch': 7.07} {'loss': 2.5886, 'grad_norm': 0.605222225189209, 'learning_rate': 1.4691650853889943e-05, 'epoch': 7.08} {'loss': 3.1832, 'grad_norm': 2.3125052452087402, 'learning_rate': 1.4655159830681653e-05, 'epoch': 7.08} {'loss': 2.4891, 'grad_norm': 1.2619099617004395, 'learning_rate': 1.4618668807473363e-05, 'epoch': 7.09} the eval loss seems to be droping but it slowed down significantly and now it bearly changes and the loss dropped a little but stopped. is it maube because there isnt enough data and the increased epochs dont have as much of a effect. in this data i have 117000 records
