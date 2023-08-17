from flask import Flask, escape, request, render_template
import pickle



vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalised_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predicts():
    news = request.form['news']
    print(news)
    predict = model.predict(vector.transform([news]))
    print(predict)
    return render_template("prediction.html", prediction_text="News Headline is -> {}".format(predict))


@app.route('/prediction', methods=['GET'])
def prediction():
   return render_template("prediction.html")
 

        
        

if __name__== '__main__':
    # app.debug = True
    app.run()
    from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('fake_news_dataset.csv')

# Vectorize the text data
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(data['text'])

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_vec, data['label'])

# Set a threshold for the confidence score
threshold = 0.5

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    request_data = request.get_json()
    text = request_data['text']
    
    # Vectorize the text data
    text_vec = vectorizer.transform([text])
    
    # Make a prediction and calculate confidence score
    pred = clf.predict(text_vec)[0]
    proba = clf.predict_proba(text_vec)[0][1]
    
    # Set the response data
    response_data = {'prediction': pred, 'confidence': proba}
    
    # Add a confidence level
    if proba >= threshold:
        response_data['confidence_level'] = 'High'
    else:
        response_data['confidence_level'] = 'Low'
    
    # Return the response
    return jsonify(response_data)

if __name__ == '_main_':
    app.run(debug=True)