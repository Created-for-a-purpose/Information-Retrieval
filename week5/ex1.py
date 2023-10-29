import tarfile
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Extract the 20 Newsgroups dataset
data_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\IR\\labreport\\week5\\20news-19997.tar.gz"
extracted_folder = "20news-19997"
if not os.path.exists(extracted_folder):
    with tarfile.open(data_path, "r:gz") as tar:
        tar.extractall()

# Step 2: Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Step 3: Preprocess the data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, newsgroups.target, test_size=0.2, random_state=42)

# Step 5: Train and test Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_f1_score = f1_score(y_test, nb_predictions, average='weighted')

# Step 6: Train and test Rocchio classifier
rocchio_classifier = NearestCentroid()
rocchio_classifier.fit(X_train, y_train)
rocchio_predictions = rocchio_classifier.predict(X_test)
rocchio_f1_score = f1_score(y_test, rocchio_predictions, average='weighted')

# Step 7: Train and test K-Nearest Neighbor classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors (K) as needed
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_f1_score = f1_score(y_test, knn_predictions, average='weighted')

# Step 8: Write the F1-scores to a file
with open("f1_scores.txt", "w") as f1_file:
    f1_file.write("Naive Bayes Classifier F1-score: {:.4f}\n".format(nb_f1_score))
    f1_file.write("Rocchio Classifier F1-score: {:.4f}\n".format(rocchio_f1_score))
    f1_file.write("K-Nearest Neighbor Classifier F1-score: {:.4f}\n".format(knn_f1_score))

# Step 9: Create a bar chart of F1-scores
classifiers = ['Naive Bayes', 'Rocchio', 'K-Nearest Neighbor']
f1_scores = [nb_f1_score, rocchio_f1_score, knn_f1_score]
colors = ['black', 'yellow', 'green']  # Blue, Green, and Red colors for the bars

plt.bar(classifiers, f1_scores, color=colors)
plt.xlabel('Classifiers')
plt.ylabel('F1-Score')
plt.title('F1-Scores for Text Classifiers')
plt.show()

print("F1-scores have been written to 'f1_scores.txt'.")
