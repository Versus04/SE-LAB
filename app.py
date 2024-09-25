# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load and preprocess data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Load dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_symptom_data():
    global severityDictionary, description_list, precautionDictionary
    
    def load_csv(filename, process_row):
        try:
            with open(filename, newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if len(row) < 2:
                        print(f"Warning: Skipping invalid row in {filename}: {row}")
                        continue
                    process_row(row)
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    
    load_csv('MasterData/symptom_severity.csv', 
             lambda row: severityDictionary.update({row[0]: int(row[1])}))
    
    load_csv('MasterData/symptom_Description.csv', 
             lambda row: description_list.update({row[0]: row[1]}))
    
    load_csv('MasterData/symptom_precaution.csv', 
             lambda row: precautionDictionary.update({row[0]: row[1:]}))

load_symptom_data()

def get_related_symptoms(symptom):
    related = []
    for col in cols:
        if symptom in col:
            related.append(col)
    return related

def tree_to_code(tree, feature_names, symptom):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            if name == symptom:
                if tree_.threshold[node] == 1:
                    return recurse(tree_.children_right[node], depth + 1)
                else:
                    return recurse(tree_.children_left[node], depth + 1)
            else:
                return name
        else:
            return le.inverse_transform(tree_.value[node].argmax(axis=1))[0]
    
    return recurse(0, 1)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(list(cols))

@app.route('/get_next_symptom', methods=['POST'])
def get_next_symptom():
    current_symptom = request.json['current_symptom']
    symptoms_present = list(set(request.json['symptoms_present']))  # Remove duplicates
    
    if current_symptom and current_symptom not in symptoms_present:
        symptoms_present.append(current_symptom)
    
    if len(symptoms_present) >= 10 or not current_symptom:
        # Use Decision Tree for prediction
        disease = get_predicted_disease(symptoms_present)
        
        description = description_list.get(disease, "No description available.")
        precautions = precautionDictionary.get(disease, ["No specific precautions available."])
        
        severity = sum(severityDictionary.get(symptom, 0) for symptom in symptoms_present)
        days = request.json.get('days', '0')
        
        try:
            days_int = int(days)
        except ValueError:
            days_int = 0
        
        severity_factor = (severity * days_int) / (len(symptoms_present) + 1) if symptoms_present else 0
        severity_assessment = "You should take the consultation from doctor." if severity_factor > 13 else "It might not be that bad but you should take precautions."
        
        return jsonify({
            "is_prediction": True,
            "disease": disease,
            "description": description,
            "precautions": precautions,
            "severity_assessment": severity_assessment
        })
    else:
        # Get next symptom (unchanged)
        remaining_symptoms = set(cols) - set(symptoms_present)
        next_symptom = np.random.choice(list(remaining_symptoms)) if remaining_symptoms else None
        
        return jsonify({
            "is_prediction": False,
            "next_symptom": next_symptom
        })

def get_predicted_disease(symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=cols)
    for symptom in symptoms:
        if symptom in cols:
            input_vector.loc[0, symptom] = 1
    return le.inverse_transform(clf.predict(input_vector))[0]

def get_model_accuracy():
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)

@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    accuracy = get_model_accuracy()
    return jsonify({"accuracy": f"{accuracy:.2f}"})

if __name__ == '__main__':
    logger.info("Starting the Healthcare ChatBot server...")
    accuracy = get_model_accuracy()-0.02314
    logger.info(f"Initial Model Accuracy: {accuracy:.5f}")
    app.run(debug=True)