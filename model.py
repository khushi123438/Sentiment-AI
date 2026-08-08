import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# 1. DATASET PATH
# =========================================================

DATASET_PATH = "aclImdb"


# =========================================================
# 2. FUNCTION TO LOAD REVIEWS
# =========================================================

def load_reviews(folder_path, sentiment):

    texts = []
    labels = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

                texts.append(text)
                labels.append(sentiment)

    return texts, labels


# =========================================================
# 3. LOAD TRAINING DATA
# =========================================================

train_pos_path = os.path.join(
    DATASET_PATH,
    "train",
    "pos"
)

train_neg_path = os.path.join(
    DATASET_PATH,
    "train",
    "neg"
)


positive_reviews, positive_labels = load_reviews(
    train_pos_path,
    "Positive"
)

negative_reviews, negative_labels = load_reviews(
    train_neg_path,
    "Negative"
)


X_train = positive_reviews + negative_reviews
y_train = positive_labels + negative_labels


print("Training reviews:", len(X_train))


# =========================================================
# 4. LOAD TEST DATA
# =========================================================

test_pos_path = os.path.join(
    DATASET_PATH,
    "test",
    "pos"
)

test_neg_path = os.path.join(
    DATASET_PATH,
    "test",
    "neg"
)


positive_test, positive_test_labels = load_reviews(
    test_pos_path,
    "Positive"
)

negative_test, negative_test_labels = load_reviews(
    test_neg_path,
    "Negative"
)


X_test = positive_test + negative_test

y_test = (
    positive_test_labels +
    negative_test_labels
)


print("Testing reviews:", len(X_test))


# =========================================================
# 5. NLP + MACHINE LEARNING PIPELINE
# =========================================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            stop_words="english",

            max_features=50000,

            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=1000
        )
    )

])


# =========================================================
# 6. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


print("Training completed!")


# =========================================================
# 7. EVALUATION
# =========================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nModel Accuracy:", accuracy)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 8. PREDICTION FUNCTION
# =========================================================

def predict_sentiment(text):

    prediction = model.predict(
        [text]
    )[0]

    probabilities = model.predict_proba(
        [text]
    )[0]

    confidence = probabilities.max()

    return (
        prediction,
        round(confidence * 100, 2)
    )