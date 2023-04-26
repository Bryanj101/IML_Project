import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Load the appliance dataset
with gzip.open('Appliances.json.gz', 'r') as f:
    df = pd.read_json(f, lines=True)
    
df.dropna(subset=['reviewText'], inplace=True)


df['review_length'] = df['reviewText'].apply(lambda x: len(x.split()))

sns.displot(df['review_length'], kde=True, bins=100)
plt.xlabel('Review text length')
plt.ylabel('Count')
plt.title('Distribution of review text length')
plt.xlim(0, 500)
plt.xticks(range(0, 500, 50))
plt.figure(figsize=(20, 10))
plt.show()