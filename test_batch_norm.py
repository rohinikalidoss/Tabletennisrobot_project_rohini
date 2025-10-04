import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
from batch_norm_pinn_model import BatchNormPINN

# Load model
model = tf.keras.models.load_model(
    'C:/tmp/nn_model/batch_norm_model/model.hdf5',
    custom_objects={'BatchNormPINN': BatchNormPINN}
)

# Load scalers (same reconstruction as training script)
with open('target_scaler.json', 'r') as f:
    target_scaler_params = json.load(f)
with open('control_scaler.json', 'r') as f:
    control_scaler_params = json.load(f)

target_scaler = MinMaxScaler()
target_scaler.scale_ = np.array(target_scaler_params['scale_'])
target_scaler.min_ = np.array(target_scaler_params['min_'])
target_scaler.data_min_ = np.array(target_scaler_params['data_min_'])
target_scaler.data_max_ = np.array(target_scaler_params['data_max_'])
target_scaler.data_range_ = np.array(target_scaler_params['data_range_'])

control_scaler = MinMaxScaler()
control_scaler.scale_ = np.array(control_scaler_params['scale_'])
control_scaler.min_ = np.array(control_scaler_params['min_'])
control_scaler.data_min_ = np.array(control_scaler_params['data_min_'])
control_scaler.data_max_ = np.array(control_scaler_params['data_max_'])
control_scaler.data_range_ = np.array(control_scaler_params['data_range_'])

# Test your problematic case
target_position = np.array([[1.7906, -0.4635, 0.7835]])
target_scaled = target_scaler.transform(target_position)
prediction_scaled = model.predict(target_scaled, verbose=0)
prediction = control_scaler.inverse_transform(prediction_scaled)

print("BatchNorm Model Results:")
print(f"Target: {target_position[0]}")
print(f"RPM1: {prediction[0][0]:.1f}, RPM2: {prediction[0][1]:.1f}, RPM3: {prediction[0][2]:.1f}")
print(f"Launch Angle: {prediction[0][3]:.2f}°, Azimuth: {prediction[0][4]:.2f}°")
print("Expected: ~10cm accuracy instead of 300cm!")
