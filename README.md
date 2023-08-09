# Python-Sentiment-Analysis-
Explore sentiment analysis on Amazon reviews using NLTK, uncovering sentiment trends and insights from customer feedback in this NLP-focused GitHub repository

# Using Text Reviews for Amazon Fine Food

### Importing Appropriate Libaries + Data Set
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

df = pd.read_csv(r'Reviews.csv')
df.head()
```
Importing the necessary libraries assists in performing data visualization, and adding reviews for amazon

```
print(df['Text'].values[0])
print(df.shape)
```
This will give us the first line of food review in the 'Text' column.

## QUICK EDA
```
ak = df['Score'].value_counts().sort_index().plot(kind='bar', title = 'Count of Reviews by Stars', figsize = (9,5))
ak.set_xlabel('Review Stars')

# This tells us the number of entries for a cetain raiting
```
![image](https://github.com/EesaK15/Python-Sentiment-Analysis-/assets/141469262/b13c2fac-5fce-4143-b4a7-6e9169ff146f)

### Splitting the words, and finding various sentances + NLTK Importing
```
# Basic NLTK
example = df['Text'].values[50]
print(example)

token = nltk.word_tokenize(example) # splits each word per sentance
token[:10]

import nltk
nltk.download('averaged_perceptron_tagger')
```
In NLTK (Natural Language Toolkit), a token refers to a sequence of characters that represent a single meaningful unit of text. Tokenization is the process of breaking down a text into individual tokens or words. 

```
nltk.pos_tag(token)
tagged = nltk.pos_tag(token)
tagged[:10]
```
The provided code utilizes NLTK's pos_tag function to assign parts of speech tags to a list of tokens, and then displays the tagged results for the first 10 tokens.

```
nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
```

This code downloads required NLTK datasets for named entity recognition and chunking, applies named entity chunking to a list of tagged tokens to identify and group named entities, and finally prints the identified named entities with their types.

### Vader Sentiment Scoring

NLTKS will give us the scores of the text
BAG OF WORDS' - Stop words are removed, and each word is scored and combined to a total score

```
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
```
The code downloads the VADER lexicon used for sentiment analysis in NLTK and initializes a SentimentIntensityAnalyzer for analyzing sentiment in text.

```
sia.polarity_scores('HAPPY') # 78.7 % of it is positive
# Compound is an aggregation of negative and posiitve

print(example)
sia.polarity_scores(example)
```
This code will explain  the overall sentiment of the word Happy. 

```
# We want to run through each row in the table, and find the polarity score

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

```
A dictionary named 'res' is being populated with sentiment polarity scores calculated using the SentimentIntensityAnalyzer for each 'Text' entry in a DataFrame, with corresponding 'Id' values as keys. The tqdm library is used to display a progress bar during the iteration process.

```
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()
```

```
ax = sns.barplot(data = vaders, x = 'Score', y = 'compound')
ax.set_title('Comound Score by Amazon Star Review')
plt.show()
```
![image](https://github.com/EesaK15/Python-Sentiment-Analysis-/assets/141469262/bc09ff78-0455-440e-92ca-cef75e0bb10d)

This will show the number of great reviews for Amazon as well as the average sentiment score. This shows the accuracy in greater sentiment and better reviews

```
fig, axs = plt.subplots(1,3, figsize = (10,3))

sns.barplot(data = vaders, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = vaders, x = 'Score', y = 'neu', ax = axs[1])
sns.barplot(data = vaders, x = 'Score', y = 'neg', ax = axs[2])

axs[0].set_title('Positive ')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

```
![image](https://github.com/EesaK15/Python-Sentiment-Analysis-/assets/141469262/3a72f578-e0c8-4e34-908e-966081b058d6)

### Roberta Pretrained Model 
#### Installation
```
pip install transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Use a model trained of a large corpus of data
# Transformer model accounts for the words but also the context related to the words
# Use a model trained of a large corpus of data
# Transformer model accounts for the words but also the context related to the words

```
```
# specific model that has been pre trained for sentiment
MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

sia.polarity_scores(example)
```
The code calculates sentiment polarity scores for the given 'example' text using the SentimentIntensityAnalyzer (SIA).
```
# Run for Roberta Model
encoded = tokenizer(example, return_tensors = 'pt')

def polarity_scores_roberta(example):
    encoded = tokenizer(example, return_tensors = 'pt')
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg':scores[0],
        'roberta_neu':scores[1],
        'roberta_pos':scores[2]}

    return scores_dict
```
### Transformers Pipeline
#### Installation + Examples
```
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


sent_pipeline('I love sentiment analysis!')
sent_pipeline('Make sure to like and subscribe!')
```






