from flask import Flask, render_template, request
from model import predict_sentiment

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    text = ""

    if request.method == "POST":

        text = request.form["text"]

        if text.strip():

            sentiment, confidence = predict_sentiment(text)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)

