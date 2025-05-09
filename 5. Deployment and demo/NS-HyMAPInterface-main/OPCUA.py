from opcua import Client

def collect_data():
    global R01_Load, R04_Load, CycleState
    server_endpoint = "opc.tcp://192.168.0.2:4840"
    client = Client(server_endpoint)
    client.connect()
    node_inputs = client.get_node("ns=3;s=Inputs")
    node_outputs = client.get_node("ns=3;s=Outputs")
    CycleState = node_outputs.get_child("Q_Cell_CycleState").get_value()
    R02_Pot = node_inputs.get_child("I_R02_Gripper_Pot").get_value()
    R03_Pot = node_inputs.get_child("I_R03_Gripper_Pot").get_value()
    print(R02_Pot)
    print(R03_Pot)
    print(CycleState)
    client.disconnect()
    return R02_Pot,R03_Pot,CycleState


collect_data()

