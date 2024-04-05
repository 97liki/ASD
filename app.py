from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS  # Import CORS
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
CORS(app) 

# Load the model
classifier = joblib.load('saved_model.pkl')  # Ensure correct path

# Define your model columns based on your previous setup
model_columns=['age',
 'result',
 'A1_Score',
 'A2_Score',
 'A3_Score',
 'A4_Score',
 'A5_Score',
 'A6_Score',
 'A7_Score',
 'A8_Score',
 'A9_Score',
 'A10_Score',
 'gender_f',
 'gender_m',
 'ethnicity_Asian',
 'ethnicity_Black',
 'ethnicity_Hispanic',
 'ethnicity_Latino',
 'ethnicity_Middle Eastern ',
 'ethnicity_Others',
 'ethnicity_Pasifika',
 'ethnicity_South Asian',
 'ethnicity_Turkish',
 'ethnicity_White-European',
 'ethnicity_others',
 'jundice_no',
 'jundice_yes',
 'austim_no',
 'austim_yes',
 'contry_of_res_Afghanistan',
 'contry_of_res_AmericanSamoa',
 'contry_of_res_Angola',
 'contry_of_res_Armenia',
 'contry_of_res_Aruba',
 'contry_of_res_Australia',
 'contry_of_res_Austria',
 'contry_of_res_Bahamas',
 'contry_of_res_Bangladesh',
 'contry_of_res_Belgium',
 'contry_of_res_Bolivia',
 'contry_of_res_Brazil',
 'contry_of_res_Burundi',
 'contry_of_res_Canada',
 'contry_of_res_Chile',
 'contry_of_res_China',
 'contry_of_res_Costa Rica',
 'contry_of_res_Cyprus',
 'contry_of_res_Czech Republic',
 'contry_of_res_Ecuador',
 'contry_of_res_Egypt',
 'contry_of_res_Ethiopia',
 'contry_of_res_Finland',
 'contry_of_res_France',
 'contry_of_res_Germany',
 'contry_of_res_Iceland',
 'contry_of_res_India',
 'contry_of_res_Indonesia',
 'contry_of_res_Iran',
 'contry_of_res_Ireland',
 'contry_of_res_Italy',
 'contry_of_res_Jordan',
 'contry_of_res_Malaysia',
 'contry_of_res_Mexico',
 'contry_of_res_Nepal',
 'contry_of_res_Netherlands',
 'contry_of_res_New Zealand',
 'contry_of_res_Nicaragua',
 'contry_of_res_Niger',
 'contry_of_res_Oman',
 'contry_of_res_Pakistan',
 'contry_of_res_Philippines',
 'contry_of_res_Portugal',
 'contry_of_res_Romania',
 'contry_of_res_Russia',
 'contry_of_res_Saudi Arabia',
 'contry_of_res_Serbia',
 'contry_of_res_Sierra Leone',
 'contry_of_res_South Africa',
 'contry_of_res_Spain',
 'contry_of_res_Sri Lanka',
 'contry_of_res_Sweden',
 'contry_of_res_Tonga',
 'contry_of_res_Turkey',
 'contry_of_res_Ukraine',
 'contry_of_res_United Arab Emirates',
 'contry_of_res_United Kingdom',
 'contry_of_res_United States',
 'contry_of_res_Uruguay',
 'contry_of_res_Viet Nam',
 'relation_Health care professional',
 'relation_Others',
 'relation_Parent',
 'relation_Relative',
 'relation_Self']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert form data to dictionary
    form_data = request.form.to_dict()
    processed_data = preprocess_data(form_data)
    
    # Make prediction
    prediction = classifier.predict(processed_data)
    result = "ASD" if prediction[0] == 1 else "No ASD"
    
    return jsonify({'result': result})

def preprocess_data(form_data):
    # Initialize data for all model features
    data = pd.DataFrame(columns=model_columns, index=[0])
    data.loc[0] = 0  # Initialize all values to 0
    
    # Numeric data
    data.loc[0, 'age'] = float(form_data['age'])
    
    # Scores and result
    for i in range(1, 11):
        data.loc[0, f'A{i}_Score'] = int(form_data.get(f'a{i}_score', 0))
    
    # Calculate result based on A1_Score to A10_Score
    data.loc[0, 'result'] = sum(data.loc[0, f'A{i}_Score'] for i in range(1, 11))
    
    # Gender
    if form_data['gender'] == 'f':
        data.loc[0, 'gender_f'] = 1
    else:
        data.loc[0, 'gender_m'] = 1
    
    # Ethnicity
    ethnicity_key = 'ethnicity_' + form_data['ethnicity'].replace(' ', '_')
    if ethnicity_key in data.columns:
        data.loc[0, ethnicity_key] = 1
    
    # Jaundice
    data.loc[0, 'jundice_yes'] = 1 if form_data['jaundice'] == 'yes' else 0
    
    # Autism in family
    data.loc[0, 'austim_yes'] = 1 if form_data['austim'] == 'yes' else 0
    
    # Country of Residence
    country_key = 'contry_of_res_' + form_data['country_of_res'].replace('', '_')
    if country_key in data.columns:
        data.loc[0, country_key] = 1
    
    # Relation
    relation_key = 'relation_' + form_data['relation'].replace(' ', '_')
    if relation_key in data.columns:
        data.loc[0, relation_key] = 1
    
    return data
if __name__ == '__main__':
    app.run(debug=True)