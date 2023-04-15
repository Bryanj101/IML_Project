from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset 
from tqdm.auto import tqdm
import torch
import pipe
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load('ML_project/ml_torch_model.pt')

pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

print(pipe("This oven is good."))