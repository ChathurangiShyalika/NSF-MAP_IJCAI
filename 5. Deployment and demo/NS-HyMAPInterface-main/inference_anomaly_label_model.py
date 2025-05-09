import xgboost as xgb
import numpy as np

loaded_model = xgb.XGBRegressor()
loaded_model.load_model('anomaly_label_model.json')

single_row_input = np.array([[11650, 2400, 200]], dtype=float)  

prediction = loaded_model.predict(single_row_input)
print(f"Prediction for the input row: {prediction}")