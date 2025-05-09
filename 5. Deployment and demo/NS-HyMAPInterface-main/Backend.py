import json
import time
from opcua import Client
import numpy as np
from flask import Flask, jsonify
import xgboost as xgb
import numpy as np
import pickle
# import Model
import threading
import os
import pandas as pd
import csv
import RCAFiles
from dotenv import load_dotenv
from RCAFiles.python_files.classes.neo4j_connection import Neo4jConnection
from RCAFiles.python_files.classes.ontology import Ontology
from RCAFiles.python_files.classes.reasoner import AnomalyReasoner
from flask_cors import CORS
from ImageCap import PylonCameras
import NS_HyMAP_main, ns_hymap_inference


# cap = PylonCameras(num_devices=2)
app = Flask(__name__)
#CORS(app)  # Enable CORS for all routes


stop_event = threading.Event()


# cap.grab('LatestOnly')
AnomalyLabel = 100
count = 0
R02 = []
R03 = []
RCAData = []
CycleState = 0
Predicted_R02 = 0
Predicted_R03 = 0
load_dotenv()
URI = 'bolt://localhost:7687'
USER = 'neo4j'
PASSWORD = 'twine-grooves-accelerations'
# AUTH = (os.getenv("NEO4J_USER_NAME"), os.getenv("NEO4J_PASSWD"))

@app.route('/get_latest', methods=['GET'])
def get_latest():
    global R02, R03, CycleState, AnomalyLabel, Predicted_R02, Predicted_R03, count
    if count == 0:
        time_series_data = [[0, 0, 100]]
        count = count +1
    else:
        time_series_data = [[R02[-1], R03[-1], AnomalyLabel]]

    data = ns_hymap_inference.prepare_time_series(time_series_data)

    outputTS = ns_hymap_inference.make_inference_woutImage(data)

    # loaded_model = xgb.XGBRegressor()
    # loaded_model.load_model('model.json')
    #
    # single_row_input = np.array(time_series_data, dtype=float)
    #
    # prediction = loaded_model.predict(single_row_input)
    # print(f"Prediction for the input row: {prediction}")
    # Update Anomaly label based of the output we just got
    AnomalyLabel = outputTS[0, 0, -1].item()

    print(f"Anomaly Value: {AnomalyLabel}")
    Predicted_R02 = outputTS[0, 0, 0].item()
    Predicted_R03 = outputTS[0, 0, 1].item()

    output_data = {
        'time_series_data': time_series_data,
        'output_with_Image': outputTS.tolist(),  # Convert tensor to list for JSON serialization
        'output_without_Image': outputTS.tolist(),  # Same for this tensor
        # 'Image Path': image_path
    }
    # Append the output data to the JSON file
    NS_HyMAP_main.append_to_json_file(output_data, NS_HyMAP_main.json_file_path)

    try:
        output = jsonify({'latest_R02': R02[-1], 'latest_R03': R03[-1],
                          'CycleState': CycleState, 'AnomalyLabel': AnomalyLabel,
                          'Predicted_R02': Predicted_R02, 'Predicted_R03': Predicted_R03})
        return output
    except IndexError:
        print("No values yet")
        return jsonify({'latest_R02': None, 'latest_R03': None, 'CycleState': "No Values", 'AnomalyLabel': None,
                        'Predicted_R02': None, 'Predicted_R03': None})


def collect_data():
    global R02, R03, CycleState, RCAData, count, AnomalyLabel
    server_endpoint = "opc.tcp://192.168.0.2:4840"
    client = Client(server_endpoint)
    client.connect()
    node_inputs = client.get_node("ns=3;s=Inputs")
    node_outputs = client.get_node("ns=3;s=Outputs")

    while not stop_event.is_set():  # Continue collecting data until stop_event is set
        try:
            CycleState = node_outputs.get_child("Q_Cell_CycleState").get_value()
            R03_Pot = node_inputs.get_child("I_R03_Gripper_Pot").get_value()
            R02_Pot = node_inputs.get_child("I_R02_Gripper_Pot").get_value()

            R02.append(R02_Pot)
            R03.append(R03_Pot)

            sensor_variables = {
                "I_R02_Gripper_Pot": R02_Pot,
                "I_R03_Gripper_Pot": R03_Pot,
                # Add more sensor variables as needed
            }
            structured_data = {
                "cycle_state": CycleState,
                "sensor_variables": sensor_variables
            }
            RCAData.append(structured_data)

            time.sleep(1)  # Add a small sleep to reduce CPU usage

        except Exception as e:
            print(f"Error collecting data: {e}")
            break  # Exit the loop if there's an error


def get_formatted_data(filepath):
    """
    Return the data in the following format
    [ {'cycle_state': <value>,
    'sensor_variables':{'I_R01_Gripper_Load':<value>,
                        'I_R02_Gripper_Load':<value>,
                        'I_R04_Gripper_Load':<value>,...}
    }, {'cycle_state':<value>, "",..}..]
    # number of dict = number of rows in csv/df
    """
    data_list = []
    df = pd.read_csv(filepath)
    df = df.drop(labels=['Description', '_time'], axis=1) #0-rows, 1-columns
    headers = df.columns.tolist()
    for i in range(0,len(df)):
        data_dict = {}
        sensor_dict = {}
        for header in headers:
            if header == 'CycleState':
                data_dict['cycle_state'] = df[header][i]
            else:
                sensor_dict[str(header)] = df[header][i]
        data_dict['sensor_variables'] = sensor_dict
        data_list.append(data_dict)

    return data_list


def save_specified_values_if_changed(df):
    saved_values = []
    seen_entries = set()

    for idx, row in df.iterrows():
        entry = (
            row['anomalous_sensor_variables'],
            row['robot_names'],
            row['cycle_function'],
            row['cycle_state'],
            row['sensor_names']
        )

        if entry not in seen_entries:
            saved_values.append({
                "anomalous_sensor_variables": row['anomalous_sensor_variables'],
                "robot_names": row['robot_names'],
                "cycle_function": row['cycle_function'],
                "cycle_state": row['cycle_state'],
                "sensor": row['sensor_names']
            })
            seen_entries.add(entry)

    return saved_values

@app.route('/RootCause', methods=['GET'])
def RootCause():
    from RCAFiles.python_files.main import get_min_max_data, get_cycle_function_data
    neo4j_obj = Neo4jConnection(uri=URI,
                                user=USER,
                                pwd=PASSWORD)
    # Specify the filepaths
    ontology_filepath = "RCAFiles/mfg-data/process_ontology.txt" # filepath that consists of ontology creation query
    min_max_filepath = 'RCAFiles/mfg-data/cycle_state_values.csv' # filepath that consists of min and max values of sensors as per cycle state
    cycle_function_path = 'RCAFiles/mfg-data/cycle_state_function.csv'
    # create an object for ontology class
    ont = Ontology()

    # Inject ontology to Neo4j when empty
    nodes = neo4j_obj.query("MATCH (n) RETURN n")

    # TODO - create constraint
    res = ont.create(neo4j_obj, ontology_filepath)
    print("Result of Ontology Creation:", res)
    # get the data in required format to update ontology
    min_max_data = get_min_max_data(filepath=min_max_filepath)
    # call the update function for min max
    res = ont.update_min_max(neo4j_obj, min_max_data)
    print("Min Max value Update:", res)

    # get the cycle functions and load it into the ontology
    cycle_function_data = get_cycle_function_data(filepath=cycle_function_path)
    # call the required function
    res = ont.add_cycle_functions(neo4j_obj, cycle_function_data)
    print("Adding Cycle Functions:", res)


    ############## ONTOLOGY USAGE ##########
    """
    The following code will generate reasoning. Now that the ontology is created, you can use 
    it for explanations
    """

    # get the data for anomalous cycle
    # NOTE: Input file must contain only the cycle state and sensor values
    # anomalous_data_filepath = 'RCAFiles/mfg-data/anomaly_data/fadi2.csv'
    # anomalous_data = get_formatted_data(anomalous_data_filepath)
    # print(RCAData)
    anomalous_data = RCAData

    # get the explanation for anomaly
    # Instantiate Reasoner class
    reasoner = AnomalyReasoner()
    exp_dict = reasoner.get_explanation(neo4j_obj, anomalous_data)
    # store the values in a csv file
    df = pd.DataFrame.from_dict(exp_dict)
    anomalies = save_specified_values_if_changed(df)
    print(anomalies)
    return anomalies


if __name__ == '__main__':
    stop_event.clear()  # Ensure the event is cleared before starting the thread
    #
    data_thread = threading.Thread(target=collect_data)
    data_thread.daemon = True
    data_thread.start()

    # Run the Flask application
    app.run(port=5002, debug=True)
