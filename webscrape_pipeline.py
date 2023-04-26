import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BertTokenizer
import re
import torch

# Load the sentiment analysis pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load('ml_torch_model.pt')
pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

# Scrape the reviews from Yelp
url = 'https://www.yelp.com/biz/gregoire-berkeley'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
regex = re.compile("comment")
reviews = soup.find_all('p', {'class': regex})

# Set the number of reviews to analyze
num_reviews = 10

# Loop through the reviews and perform sentiment analysis
for i, review in enumerate(reviews):
    if i >= num_reviews:
        break
    text = review.get_text()
    output = pipe(text)
    label = output[0]['label']
    score = output[0]['score']
    # Print the results
    if label == 'LABEL_5':
        print("{}\n\nThe sentiment of the input is very positive with a confidence score of {:.2f}.".format(text, score))
    elif label == 'LABEL_4':
        print("{}\n\nThe sentiment of the input is positive with a confidence score of {:.2f}.".format(text, score))
    elif label == 'LABEL_3':
        print("{}\n\nThe sentiment of the input is neutral with a confidence score of {:.2f}.".format(text, score))
    elif label == 'LABEL_2':
        print("{}\n\nThe sentiment of the input is negative with a confidence score of {:.2f}.".format(text, score))
    else:
        print("{}\n\nThe sentiment of the input is very negative with a confidence score of {:.2f}.".format(text, score))