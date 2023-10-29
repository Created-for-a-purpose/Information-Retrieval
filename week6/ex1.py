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
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

feature_names = vectorizer.get_feature_names_out()
output_file = "output.txt"

# Redirecting standard output to a file
import sys
original_stdout = sys.stdout
with open(output_file, 'w') as f:
    sys.stdout = f
    print_top_words(svd.components_, feature_names)
# Restoring the standard output
sys.stdout = original_stdout

# Information Retrieval

# You can now make any edits to the code or use the "output.txt" file as needed.


# @title
query = "computer science"
preprocessed_query = preprocess_text(query)
query_vector = vectorizer.transform([preprocessed_query])
query_lsa = svd.transform(query_vector)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(query_lsa, lsa_matrix)
most_similar_doc_indices = cosine_similarities[0].argsort()[::-1]

top_n = 5  # Number of most relevant documents to retrieve
most_similar_docs = [documents[i] for i in most_similar_doc_indices[:top_n]]

# Evaluation
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Example of clustering using K-means
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(lsa_matrix)

# Example of purity evaluation
from collections import Counter

def calculate_purity(actual_labels, cluster_labels):
    label_counts = Counter(actual_labels)
    total_samples = len(actual_labels)
    purity = 0
    for label, count in label_counts.items():
        purity += (count / total_samples) * max([
            np.sum((actual_labels == label) & (cluster_labels == cluster_label))
            for cluster_label in np.unique(cluster_labels)
        ])
    return purity

purity = calculate_purity(newsgroups.target, cluster_labels)

# Example of NMI and Silhouette evaluation
nmi = normalized_mutual_info_score(newsgroups.target, cluster_labels)
silhouette = silhouette_score(lsa_matrix, cluster_labels)

print(f"Purity: {purity}")
print(f"Normalized Mutual Information: {nmi}")
print(f"Silhouette Score: {silhouette}")