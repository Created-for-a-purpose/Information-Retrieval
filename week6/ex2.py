# Data Collection
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
actual_labels = newsgroups.target_names

# Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

preprocessed_documents = [preprocess_text(doc) for doc in documents]


# Create Term-Document Matrix using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
term_doc_matrix = vectorizer.fit_transform(preprocessed_documents)

# SVD Decomposition
from sklearn.decomposition import TruncatedSVD

num_topics = 100
svd = TruncatedSVD(n_components=num_topics)
lsa_matrix = svd.fit_transform(term_doc_matrix)


# Topic Exploration
import numpy as np

def print_top_words(components, feature_names, n_words=10):
    for topic_idx, topic in enumerate(components):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        # print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

feature_names = vectorizer.get_feature_names_out()


# Information Retrieval

# Print the headings of the top 5 documents
# Information Retrieval
query = "Car Engine"
preprocessed_query = preprocess_text(query)
query_vector = vectorizer.transform([preprocessed_query])
query_lsa = svd.transform(query_vector)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(query_lsa, lsa_matrix)
most_similar_doc_indices = cosine_similarities[0].argsort()[::-1]

# Retrieve the top 5 documents
top_n = 7
most_similar_docs = [newsgroups.data[i] for i in most_similar_doc_indices[:top_n]]

# Print the headings and document numbers of the top 5 documents
for doc_idx, doc in enumerate(most_similar_docs[2:7]):
    category = newsgroups.target_names[newsgroups.target[most_similar_doc_indices[doc_idx]]]
    content = doc.split('\n')[0]
    print(f"Doc {doc_idx + 1}: {category}: {content}")

# Evaluation