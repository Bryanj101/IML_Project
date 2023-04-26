from transformers import pipeline, BertTokenizer
import torch
import pipe

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load('ml_torch_model.pt')

pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

output = pipe("Will not run my device.. Even when i put the device in the direct sun doesnt work for me. It has been 16 weeks or so thought i would try twice and nope never seemed to work at all.. Now they can go right in the trash.... Thanks a bunch. Would not recommend!")

label = output[0]['label']
score = output[0]['score']

if label == 'LABEL_5':
    print("The sentiment of the input is very positive with a confidence score of {:.2f}.".format(score))
elif label == 'LABEL_4':
    print("The sentiment of the input is positive with a confidence score of {:.2f}.".format(score))
elif label == 'LABEL_3':
    print("The sentiment of the input is neutral with a confidence score of {:.2f}.".format(score))
elif label == 'LABEL_2':
    print("The sentiment of the input is negative with a confidence score of {:.2f}.".format(score))
else:
    print("The sentiment of the input is very negative with a confidence score of {:.2f}.".format(score))



