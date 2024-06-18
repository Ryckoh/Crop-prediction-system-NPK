from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)

# Load the model and the fitted LabelEncoder
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Assuming you have a dataset or a variable X with feature names
# Update this part based on your actual dataset
feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = pd.DataFrame(columns=feature_order)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and maintain the specified order
        features = [float(request.form.get(feature)) for feature in feature_order]
        input_data = pd.DataFrame([features], columns=feature_order)

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Convert the prediction back to the original label using the fitted LabelEncoder
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f'The predicted crop is: {predicted_crop}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)


