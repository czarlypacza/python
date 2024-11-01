# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the model and tokenizer
# model_path = './model2/fine_tuned_model'
# tokenizer_path = './model2/tokenizer'

# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# model.to(device)
# model.eval()

# def make_prediction(input_text):
#     encoding = tokenizer(input_text, return_tensors="pt")
#     encoding = {k: v.to(device) for k, v in encoding.items()}

#     # Perform inference
#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs.logits
#         predictions = torch.softmax(logits, dim=-1)

#     # Extract scores
#     pos_score = predictions[0][1].item()
#     neg_score = predictions[0][0].item()
#     return {'positive': pos_score, 'negative': neg_score}

# # Interactive loop for user input
# loop = True
# while loop:
#     input_text = input("Please enter your input text: ")
#     if input_text.lower() == "exit":
#         loop = False
#         break
#     prediction = make_prediction(input_text)
#     print(prediction)

# import os
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # # Disable tokenizers parallelism warning
# # os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# # Load the fine-tuned model and tokenizer
# model_path = "./results3/checkpoint-39822"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, problem_type='regression').to("cuda")

# def analyze_sentiment():
#     while True:
#         text = input("Enter text for sentiment analysis (type 'exit' to quit): ")
#         if text.lower() == 'exit':
#             break

#         # Tokenize the input text
#         inputs = tokenizer(text, return_tensors="pt").to("cuda")

#         # Get model predictions
#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Extract logits
#         logits = outputs.logits.cpu().numpy()

#         # Interpret the logits as pos and neg scores
#         pos_score = logits[0][0]
#         neg_score = logits[0][1]

#         print(f"Positive score: {pos_score}, Negative score: {neg_score}")

# if __name__ == "__main__":
#     analyze_sentiment()


# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load the fine-tuned model and tokenizer
# model_path = "./results4/checkpoint-66370"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, problem_type='regression').to("cuda")

# def analyze_sentiment(text):
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt").to("cuda")

#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Extract logits
#     logits = outputs.logits.cpu().numpy()

#     # Interpret the logits as pos and neg scores
#     pos_score = logits[0][0]
#     neg_score = logits[0][1]

#     return pos_score, neg_score

# def run_tests():
#     # Fake company reviews
#     test_reviews = [
#         "This company provides excellent service and support.",
#         "I had a terrible experience with this company.",
#         "The product quality is outstanding and the customer service is top-notch.",
#         "I would not recommend this company to anyone.",
#         "The staff are friendly and very helpful.",
#         "The service was slow and the staff were rude.",
#         "I am extremely satisfied with my purchase.",
#         "This company is a scam, avoid at all costs.",
#         "Great value for money and fast delivery.",
#         "The worst customer service I have ever experienced."
#     ]

#     # Create test_results folder if it doesn't exist
#     os.makedirs("test_results2", exist_ok=True)

#     # Open a file to save the results
#     with open("test_results2/results.txt", "w") as f:
#         for review in test_reviews:
#             pos_score, neg_score = analyze_sentiment(review)
#             sentiment = "Positive" if pos_score > neg_score else "Negative"
#             confidence = abs(pos_score - neg_score)
#             result = f"Review: {review}\nSentiment: {sentiment}, Confidence: {confidence:.4f}\nPositive score: {pos_score}, Negative score: {neg_score}\n\n"
#             f.write(result)
#             print(result)

# if __name__ == "__main__":
#     run_tests()

import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import json
from torch import nn

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

def load_model(model_path, tokenizer_path):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load the model
    model = BertForSentimentRegression.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

    return model, tokenizer

def predict(model, tokenizer, texts, max_len=256):
    # Tokenize the texts
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Make predictions
    with torch.no_grad():
        pos, neg = model(input_ids=input_ids, attention_mask=attention_mask)

    return pos, neg

if __name__ == "__main__":
    # Paths to the model and tokenizer
    model_path = "./results7/checkpoint-37715"
    tokenizer_path = "bert-base-uncased"

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)

    # Example texts for prediction
    texts = [
        "I love this product!",
        "horrible never buy again, waste of money",
    ]

    # Make predictions
    pos, neg = predict(model, tokenizer, texts)

    # Print the results
    for i, text in enumerate(texts):
        print(f"Text: {text}")
        print(f"Positive sentiment score: {pos[i].item()}")
        print(f"Negative sentiment score: {neg[i].item()}")
        print()