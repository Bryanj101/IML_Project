import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, 0)
# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = torch.load('ML_project\ml_train_dataset.pt')
val_dataset = torch.load('ML_project\ml_val_dataset.pt')
test_dataset = torch.load('ML_project\ml_test_dataset.pt')
# Instantiate the BERT model
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=6, label_first=False)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
model.cuda()

# Define the optimizer and loss function for training the model
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the batch size for training and evaluation
batch_size = 4

# Create PyTorch data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Train the model for a specified number of epochs
num_epochs = 2
for epoch in range(num_epochs):
    # Train the model on the training dataset
    model.train()
    for batch in train_loader:
        # Extract the input features and labels from the batch
        input_ids, attention_mask, labels = batch

        # Move the input tensors to the GPU
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()

        # Forward pass through the model
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        # Backward pass through the model and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation dataset
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_loss = 0
        for batch in val_loader:
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
            total_correct += torch.sum(predicted_labels == labels)
            total_samples += len(labels)

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(val_loader)
        print(f'Epoch {epoch + 1} | Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}')

# Save the trained model
torch.save(model, 'ML_project/ml_torch_model.pt')

model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
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
        total_correct += torch.sum(predicted_labels == labels)
        total_samples += len(labels)

    accuracy = total_correct / total_samples
    avg = total_loss / len(test_loader)
    print(f'Test Loss: {avg:.4f} | Test Acc: {accuracy:.4f}')


