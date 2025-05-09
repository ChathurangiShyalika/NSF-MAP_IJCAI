from flask import Flask, render_template, request, jsonify
import numpy as np
import requests


app = Flask(__name__, static_url_path='/static')

CycleState = 0
r02 = 0
r03 = 0
R02 = []
R03 = []
AnomalyLabel = None
predicting = False
@app.route('/TSF.html')
def TSC():
    # data = np.load("PotData.npy")
    # data_list = data.tolist()
    # latest_value = data_list[-1]
    return render_template('TSF.html')

@app.route('/NS_HyMAP.html')
def NS_HyMAP():
    return render_template('NS_HyMAP.html')

@app.route('/RCA.html')
def RCA():
    return render_template('RCA.html')

@app.route('/RootCause')
def RootCause():
    response = requests.get("http://localhost:5002/RootCause")
    return response.json()

@app.route('/get_latest')
# Assume `get_latest_data()` is a function that returns the latest values of R02 and R03
def get_latest_data():
    global r02, r03, Predicted_R03, Predicted_R02
    response = requests.get("http://localhost:5002/get_latest")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the latest_R02 value from the JSON responsep
        CycleState = response.json()['CycleState']
        latest_R02 = response.json()['latest_R02']
        latest_R03 = response.json()['latest_R03']
        AnomalyLabel = response.json()['AnomalyLabel']
        Predicted_R02 = response.json()['Predicted_R02']
        Predicted_R03 = response.json()['Predicted_R03']

        return latest_R02, latest_R03, CycleState, AnomalyLabel, Predicted_R02, Predicted_R03
    else:
        # Print an error message if the request was unsuccessful
        print("Error:", response.status_code)
        return [0, 0, "N/A", "N/A", "N/A", "N/A"]  # Placeholder for demonstration purposes


@app.route('/api/data')
def api_data():
    global r02, r03, CycleState, AnomalyLabel, Predicted_R02, Predicted_R03
    r02, r03, CycleState, AnomalyLabel, Predicted_R02, Predicted_R03 = get_latest_data()
    return jsonify({'r02': r02, 'r03': r03, 'CycleState': CycleState, 'AnomalyLabel': AnomalyLabel,
                    'Predicted_R02': Predicted_R02, 'Predicted_R03': Predicted_R03})



if __name__ == '__main__':
    app.run(debug=True)
