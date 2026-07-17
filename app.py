from flask import Flask, render_template, request
import sentiment_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    sentiment = sentiment_model.predict_sentiment_vader(sentence)
    return render_template('results.html', sentence=sentence, sentiment=sentiment)

if __name__ == '__main__':
    app.run(port=5000)
