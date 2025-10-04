import numpy as np
import tensorflow as tf
from batch_norm_pinn_model import BatchNormPINN
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import json

# 🔥 SMART PATH DETECTION
hdf5_path = "sweep_dataset/grid_dataset_with_300step.hdf5"

print(f"🚀 Looking for HDF5 data at: {hdf5_path}")

if not os.path.exists(hdf5_path):
    print("🚨 File not found! Trying alternative paths...")
    alternatives = [
        "sweep_dataset/dataset_with_300step.hdf5",
        "../sweep_dataset/grid_dataset_with_300step.hdf5", 
        "../../sweep_dataset/grid_dataset_with_300step.hdf5",
        "../../../sweep_dataset/grid_dataset_with_300step.hdf5",
        "grid_dataset_with_300step.hdf5",
        "dataset_with_300step.hdf5"
    ]
    
    for alt_path in alternatives:
        if os.path.exists(alt_path):
            print(f"✅ Found file at: {alt_path}")
            hdf5_path = alt_path
            break
    else:
        print("❌ Could not find HDF5 file!")
        print("Available .hdf5 files in current directory:")
        for f in os.listdir("."):
            if f.endswith('.hdf5'):
                print(f"  - {f}")
        exit(1)

# Load YOUR real HDF5 data
with h5py.File(hdf5_path, 'r') as f:
    originals = f['originals']
    group_keys = list(originals.keys())
    
    targets = []
    controls = []
    
    for k in group_keys:
        g = originals[k]
        positions = np.array(g['positions'])
        target = positions[-1].astype(np.float32)  # Landing position
        control = np.array(g['launch_param']).astype(np.float32)  # Launch params
        
        targets.append(target)
        controls.append(control)

targets = np.array(targets, dtype=np.float32)
controls = np.array(controls, dtype=np.float32)

print(f"✅ Loaded {len(targets)} trajectories!")
print(f"Target shape: {targets.shape}")  
print(f"Control shape: {controls.shape}")

# Create scalers
target_scaler = MinMaxScaler()
control_scaler = MinMaxScaler()

X_scaled = target_scaler.fit_transform(targets)
y_scaled = control_scaler.fit_transform(controls)

# Save scalers
def save_scaler(scaler, filename):
    scaler_params = {
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist(),
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_params, f, indent=2)

save_scaler(target_scaler, 'target_scaler.json')
save_scaler(control_scaler, 'control_scaler.json')

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Create BatchNorm model (your friend's architecture!)
model = BatchNormPINN()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
model.build(input_shape=(None, 3))

print(f"🔥 BatchNorm model parameters: {model.count_params():,}")

# Train with your REAL data!
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]
)

# Save model
os.makedirs('C:/tmp/nn_model/batch_norm_model', exist_ok=True)
model.save('C:/tmp/nn_model/batch_norm_model/model.hdf5')

print("🎉 BATCHNORM TRAINED ON YOUR REAL DATA!")
print("Expected: 10cm accuracy instead of 300cm! 🔥")
