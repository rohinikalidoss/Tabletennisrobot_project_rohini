import sys
import os
import json
import logging
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
nested_package_dir = os.path.join(current_dir, 'aimy_target_shooting')
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, nested_package_dir)
from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.target_shooting_nn import TargetShootingNN
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def train_and_test_ball_launcher_mlp():
    print("TRAINING & TESTING BALL LAUNCHER MLP MODEL")
    print("=" * 50)
    config_path = get_config_path("learning")
    with open(config_path, "r") as file:
        config = json.load(file)
    print(" Configuration loaded")

    training_data_path = pathlib.Path(
        r"C:\Users\rohin\Downloads\mujoco-3.3.5-windows-x86_64\model\tabletennis_table\sweep_dataset\grid_dataset_with300step.hdf5"
    )

    if not training_data_path.exists():
        print(f"Training data not found: {training_data_path}")
        return

    print(f" Training data found: {training_data_path}")

    nn = TargetShootingNN(config, verbose=True)

    print(" Generating dataset...")
    nn.generate_dataset(filepath=training_data_path)
    print("\n=== TRAINING DATA ANALYSIS ===")
    print(f"Total training samples: {len(nn.control_parameters_norm)}")
    print("target scaler info:",vars(nn.target_scaler))
    print("control scaler info:",vars(nn.control_scaler))
    print(f"Dataset generated - Input shape: {nn.input_shape}, Output shape: {nn.output_shape}")
    target_variables_raw = nn.target_scaler.unscale(nn.target_variables_norm)
    print("\nTarget variable ranges:")
    for i in range(min(target_variables_raw.shape[1], 7)):
      print(f"  Feature {i}: {np.min(target_variables_raw[:,i]):.3f} to {np.max(target_variables_raw[:,i]):.3f}")

    test_input = np.array([2.0,0,0.76])
    print(f"\nYour test input {test_input}:")
    for i in range(3):
       in_range = np.min(target_variables_raw[:,i]) <= test_input[i] <= np.max(target_variables_raw[:,i])
    print(f"  Feature {i}: {'IN RANGE' if in_range else ' OUT OF RANGE'}")

    X = nn.target_variables_norm
    y = nn.control_parameters_norm

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Store training data back in the NN object so train_model() works
    nn.target_variables_norm = X_train
    nn.control_parameters_norm = y_train

    # Create MLP model
    print("Creating MLP model...")
    nn.generate_MLP_model()
    print("MLP model architecture created")

    # Train model
    print(" Training model...")
    nn.train_model()
    print(" Model training completed")

    # Evaluate on test data
    print(" Evaluating model on test data...")
    loss, mae = nn.model.evaluate(X_test, y_test, verbose=0)
    print(f" Test Loss: {loss:.4f}")
    print(f" Test MAE: {mae:.4f}")

    
    print(" Exporting model...")
    model_dir = pathlib.Path("/tmp/nn_model/model10/")
    model_dir.mkdir(parents=True, exist_ok=True)
    export_path = model_dir / "model10.hdf5"

    nn.export_model(export_path=export_path)
    nn.export_scaling(export_path=str(model_dir))

    print(f" Model saved to: {export_path}")
    print(" Model is ready for use in launch_ball_nn.py")

if __name__ == "__main__":
    setup_logging()
    np.set_printoptions(suppress=True)
    train_and_test_ball_launcher_mlp()