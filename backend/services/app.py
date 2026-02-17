from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
import data_process

current_file_dir = os.path.dirname(os.path.realpath(__file__)) #Get active file directory
parent_dir = os.path.dirname(current_file_dir) #Get parent of active file directory
os.chdir(os.path.dirname(parent_dir)) #Set active directory to parent of active file directory, sets directory to backend folder

loaded_model = joblib.load('backend/model/trained_model.joblib')

#Python dictionary to hold key value pairs for the reponse sent to the client currently empy defaults, acts as in memory storage holding the last results
res = {"Illness": ""}

resp_illness = []

app = Flask(__name__) #Create instance of flask app

@app.route('/predict', methods=['POST']) #Declare post route called predict to predict the data
def add_item():
    #Get JSON data from the request body
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    value_list = list(data.values()) 
    if len(value_list) != 5:
        return jsonify({"error": "Missing Fields"}), 400
    
    numeric_values = data_process.data_processing(value_list) #Converts human readable string information into numeric data the model expects
        
    pred_arr = np.array([numeric_values]) #Convert the list into an array of one element that contains all values

    try:
        prediction = loaded_model.predict(pred_arr)
        print(type(prediction))
        res = {"Illness": prediction[0]}
        return jsonify(res), 201
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__': #Launch the flask api with debugging set to true
    app.run(debug=True)