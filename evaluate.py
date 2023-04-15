import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
torch.cuda.set_per_process_memory_fraction(0.8, 0)
# Load the PyTorch datasets
train_dataset = torch.load('ML_project\ml_train_dataset.pt')
val_dataset = torch.load('ML_project\ml_val_dataset.pt')
test_dataset = torch.load('ML_project\ml_test_dataset.pt')

# Instantiate the BERT model
model = BertForSequenceClassification.from_pretrained('ML_project\ml_bert_model', num_labels=5)
model.cuda()

# Define the batch size for training and evaluation
batch_size = 4

# Create PyTorch data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=0,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    metric_for_best_model='accuracy',
    save_strategy='epoch',
    )

# Instantiate the Trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,
    eval_dataset=val_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([item[0] for item in data]),
                                'attention_mask': torch.stack([item[1] for item in data]),
                                'labels': torch.stack([item[2] for item in data])},
    compute_metrics=lambda pred: {'accuracy': (pred.predictions.argmax(axis=1) == pred.label_ids).mean()}
)

# Train the model
#trainer.train()

# Evaluate the model on the test set
print(trainer.evaluate(test_dataset))