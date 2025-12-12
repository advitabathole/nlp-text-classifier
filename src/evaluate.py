from sklearn.metrics import classification_report
import joblib

def main():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    texts = [
        "I absolutely loved it!",
        "This movie was terrible."
    ]

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    for t, p in zip(texts, preds):
        print(f"{t} → {p}")

if __name__ == "__main__":
    main()

