from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample dataset
messages = [
    "Congratulations! You have won a free lottery ticket",
    "Hey, are we meeting today?",
    "Claim your free prize now",
    "Can you send me the notes?"
]

labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(messages, labels)

text = input("Enter SMS message: ")
prediction = model.predict([text])

if prediction[0] == 1:
    print("Prediction: SPAM message")
else:
    print("Prediction: NOT SPAM")
