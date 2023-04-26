import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Clear the GPU cache
torch.cuda.empty_cache()

# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = torch.load('ml_train_dataset.pt')
val_dataset = torch.load('ml_val_dataset.pt')
test_dataset = torch.load('ml_test_dataset.pt')

# Instantiate the BERT model
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=6, label_first=False, trainable=True, max_length = 128)
model = BertForSequenceClassification.from_pretrained('ml_torch_model_state_dict', config=config)
model.cuda()

# Define the batch size for training and evaluation
batch_size = 32

# Create PyTorch data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the loss function for training the model
loss_fn = torch.nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    total_loss = 0
    for batch in test_loader:
        # Extract the input features and labels from the batch
        input_ids, attention_mask, labels = batch

        # Move the input tensors to the GPU
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()

        # Forward pass through the model
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        # Compute the accuracy of the model predictions
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.3f} | Test Acc: {accuracy:.3f}')
