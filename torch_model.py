import torch
from transformers import BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score

# Clear the GPU cache
torch.cuda.empty_cache()

# Load the dataset
train_dataset = torch.load('ml_train_dataset.pt')
val_dataset = torch.load('ml_val_dataset.pt')
test_dataset = torch.load('ml_test_dataset.pt')

# Instantiate the BERT model
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=6, label_first=False, trainable=True, max_length=128)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
model.cuda()

# Define the batch size for training and evaluation
batch_size = 32

# Create PyTorch data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Train the model for a specified number of epochs
num_epochs = 3
steps = len(train_loader) * num_epochs
learning_rate = 2e-5
print(f"Total number of training steps: {steps}")

# Define the optimizer and loss function for training the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=steps, epochs=num_epochs)

start_time = time.time()
for epoch in range(num_epochs):
    start_train_time = time.time()
    batch_num = 0
    # Train the model on the training dataset
    model.train()
    for batch in train_loader:
        batch_num += 1
        
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
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if batch_num % 50 == 0:
            elapsed_time = (time.time() - start_train_time)
            print(f"Elapsed time for {batch_num} batches: {(elapsed_time// 60):.2f} minutes {(elapsed_time % 60):.2f} seconds")

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
        print(f'Epoch {epoch + 1} | Val Loss: {avg_loss:.3f} | Val Acc: {accuracy:.3f}')

end_time = time.time()
# Save the trained model
torch.save(model, 'ml_torch_model.pt')
torch.save(model.state_dict(), 'ml_torch_model_state_dict')


# Evaluate the model on the test dataset
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
    print(f'Total time taken: {(end_time - start_time)// 60:.2f} minutes {(end_time - start_time) % 60:.2f} seconds')

