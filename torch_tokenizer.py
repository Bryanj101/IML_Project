import torch
from transformers import BertTokenizer
import pandas as pd
import json
import gzip

# Load the appliance dataset
with gzip.open('ML_project\Appliances.json.gz', 'r') as f:
    df = pd.read_json(f, lines=True, nrows=100000)

# Keep only the 'overall' and 'reviewText' columns
df = df[['overall', 'reviewText']]

# Drop rows with missing values
df.dropna(subset=['reviewText'], inplace=True)

# Map the 'overall' column to sentiment labels
sentiment_to_label = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['sentiment'] = df['overall'].map(sentiment_to_label)

# Split the dataset into training, validation, and test sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
val_df = test_df.sample(frac=0.5, random_state=42)
test_df = test_df.drop(val_df.index)

# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data and convert it to input features for BERT
train_encodings = tokenizer(list(train_df['reviewText']), truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(list(val_df['reviewText']), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(test_df['reviewText']), truncation=True, padding=True, return_tensors='pt')

# Convert the sentiment labels to numerical values (0 to 4)
train_labels = torch.tensor(list(train_df['sentiment']))
val_labels = torch.tensor(list(val_df['sentiment']))
test_labels = torch.tensor(list(test_df['sentiment']))

# Create PyTorch datasets from the input features and labels
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

# Save the PyTorch datasets
torch.save(train_dataset, 'ML_project\ml_train_dataset.pt')
torch.save(val_dataset, 'ML_project\ml_val_dataset.pt')
torch.save(test_dataset, 'ML_project\ml_test_dataset.pt')

