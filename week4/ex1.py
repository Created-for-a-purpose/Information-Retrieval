from sklearn.datasets import fetch_20newsgroups
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import json

# Fetch the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Tokenize, stem, and remove stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Preprocess the documents
preprocessed_documents = [preprocess_text(doc) for doc in newsgroups.data]

# Connect to Elasticsearch (assuming it's running on localhost)
es = Elasticsearch([{'scheme':'http' ,'host': 'localhost', 'port': 9200}])

# Function to prepare the data for bulk indexing
def prepare_data(documents):
    for i, doc in enumerate(documents):
        yield {
            "_index": "newsgroups_index",  # Change to your desired index name
            "_id": i,
            "_source": {
                "text": doc
            }
        }

# Bulk index the preprocessed dataset into Elasticsearch
bulk(es, prepare_data(preprocessed_documents))

print("Data indexed successfully.")