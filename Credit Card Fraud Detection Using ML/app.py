import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

# Load data
file_path = '/Users/raghavmehta/Desktop/Coding/python/new/Credit Card fraud detection/creditcard.csv'
data = pd.read_csv(file_path)

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html', train_acc=train_acc, test_acc=test_acc)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input feature values from form
    input_string = request.form['features']
    input_features = [float(x) for x in input_string.split('\t')]
    features = np.array(input_features, dtype=np.float64).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features)
    result = "Legitimate transaction" if prediction[0] == 0 else "Fraudulent transaction"
    return render_template('index.html', prediction_text=result, train_acc=train_acc, test_acc=test_acc)

if __name__ == "__main__":
    app.run(debug=True)
