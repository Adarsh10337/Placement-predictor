import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)
model = RandomForestClassifier()
scaler = MinMaxScaler()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('app.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def train_model():
    try:
        data = pd.read_csv('train.csv')
        data.fillna(0, inplace=True)
        label_encoder = LabelEncoder()

        # Apply label encoding to categorical columns
        categorical_columns = ['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'specialisation', 'status', 'workex']
        for column in categorical_columns:
            data[column] = label_encoder.fit_transform(data[column])

        # Scale numeric columns
        numeric_columns = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        features = ['ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'etest_p', 'specialisation', 'mba_p']
        X = data[features]
        y = data['status']

        model.fit(X, y)
        logger.info('Model training completed successfully.')
    except Exception as e:
        logger.error('An error occurred during model training: %s', str(e))

# Train the model and fit the scaler before running the Flask app
train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        sample_input = pd.DataFrame({
            'ssc_p': [float(form_data['ssc_p'])],
            'ssc_b': [form_data['ssc_b']],
            'hsc_p': [float(form_data['hsc_p'])],
            'hsc_b': [form_data['hsc_b']],
            'hsc_s': [form_data['hsc_s']],
            'degree_p': [float(form_data['degree_p'])],
            'degree_t': [form_data['degree_t']],
            'etest_p': [float(form_data['etest_p'])],
            'specialisation': [form_data['specialisation']],
            'mba_p': [float(form_data['mba_p'])]
        })

        # Apply label encoding to categorical columns in sample input
        label_encoder = LabelEncoder()
        categorical_columns = ['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'specialisation']
        for column in categorical_columns:
            sample_input[column] = label_encoder.fit_transform(sample_input[column])

        # Scale numeric columns in sample input
        numeric_columns = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
        sample_input[numeric_columns] = scaler.transform(sample_input[numeric_columns])

        prediction = model.predict(sample_input)

        result = 'Placed' if prediction[0] == 1 else 'Not Placed'
        logger.info('Prediction result: %s', result)

        return render_template('result.html', result=result)
    except Exception as e:
        logger.error('An error occurred during prediction: %s', str(e))

if __name__ == '__main__':
    app.run(debug=True)
