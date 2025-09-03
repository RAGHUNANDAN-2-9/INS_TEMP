import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load all temperature datasets
def load_all_datasets(temperatures):
    datasets = {}
    for temp in temperatures:
        try:
            df = pd.read_csv(f'mems_imu_calibration_{temp}C.csv')
            datasets[temp] = df
            print(f"Loaded dataset for {temp}°C with {len(df)} samples")
        except FileNotFoundError:
            print(f"Dataset for {temp}°C not found. Generating it...")
            from mems_imu_generator import MEMS_IMU_Calibration_Generator
            generator = MEMS_IMU_Calibration_Generator(temperature=temp, duration=3600)  # 1 hour for speed
            df = generator.generate_sensor_data()
            df.to_csv(f'mems_imu_calibration_{temp}C.csv', index=False)
            datasets[temp] = df
    return datasets

# Prepare data for training
def prepare_data_for_training(datasets, sequence_length=100):
    all_features = []
    all_targets = []
    
    for temp, df in datasets.items():
        # Use only a subset of each dataset to balance temperature representation
        subset_size = min(50000, len(df))  # Use at most 50k samples from each temperature
        df = df.iloc[:subset_size]
        
        # Features: raw sensor readings and temperature
        features = df[['raw_accel_x', 'raw_accel_y', 'raw_accel_z', 
                      'raw_gyro_x', 'raw_gyro_y', 'raw_gyro_z', 
                      'temperature']].values
        
        # Targets: error parameters to predict
        targets = df[['accel_bias_x', 'accel_bias_y', 'accel_bias_z',
                     'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z']].values
        
        # Create sequences for LSTM
        for i in range(len(features) - sequence_length):
            all_features.append(features[i:i+sequence_length])
            all_targets.append(targets[i+sequence_length])
    
    return np.array(all_features), np.array(all_targets)

# Create MLP model
def create_mlp_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6)  # Output: 6 error parameters
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse', 
                 metrics=['mae'])
    return model

# Create LSTM model
def create_lstm_model(sequence_length, input_dim):
    model = Sequential([
        Input(shape=(sequence_length, input_dim)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(6)  # Output: 6 error parameters
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse', 
                 metrics=['mae'])
    return model

# Main execution
def main():
    temperatures = [-40, -20, 0, 20, 40, 60]
    
    # Load all datasets
    print("Loading datasets...")
    datasets = load_all_datasets(temperatures)
    
    # Prepare data for training
    print("Preparing data for training...")
    sequence_length = 50  # Use 50 previous samples to predict next error
    X, y = prepare_data_for_training(datasets, sequence_length)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For MLP, we need to flatten the sequences
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_test_flat_scaled = scaler.transform(X_test_flat)
    
    # Scale the targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # For LSTM, we need to scale the features differently
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    lstm_scaler = StandardScaler()
    X_train_reshaped_scaled = lstm_scaler.fit_transform(X_train_reshaped)
    X_test_reshaped_scaled = lstm_scaler.transform(X_test_reshaped)
    
    X_train_lstm = X_train_reshaped_scaled.reshape(X_train.shape)
    X_test_lstm = X_test_reshaped_scaled.reshape(X_test.shape)
    
    # Create and train MLP model
    print("Training MLP model...")
    mlp_model = create_mlp_model(X_train_flat_scaled.shape[1])
    mlp_history = mlp_model.fit(
        X_train_flat_scaled, y_train_scaled,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # Create and train LSTM model
    print("Training LSTM model...")
    lstm_model = create_lstm_model(sequence_length, X_train.shape[2])
    lstm_history = lstm_model.fit(
        X_train_lstm, y_train_scaled,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # Evaluate models
    print("Evaluating models...")
    mlp_pred_scaled = mlp_model.predict(X_test_flat_scaled)
    mlp_pred = target_scaler.inverse_transform(mlp_pred_scaled)
    
    lstm_pred_scaled = lstm_model.predict(X_test_lstm)
    lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled)
    
    # Calculate errors
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
    lstm_mae = mean_absolute_error(y_test, lstm_pred)
    
    print(f"MLP RMSE: {mlp_rmse:.6f}, MAE: {mlp_mae:.6f}")
    print(f"LSTM RMSE: {lstm_rmse:.6f}, MAE: {lstm_mae:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_history.history['loss'], label='MLP Training Loss')
    plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
    plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
    plt.title('Model Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_history.history['mae'], label='MLP Training MAE')
    plt.plot(mlp_history.history['val_mae'], label='MLP Validation MAE')
    plt.plot(lstm_history.history['mae'], label='LSTM Training MAE')
    plt.plot(lstm_history.history['val_mae'], label='LSTM Validation MAE')
    plt.title('Model Training MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Plot predictions vs actual for a subset of test data
    plt.figure(figsize=(15, 10))
    
    # Select a random sequence to visualize
    seq_idx = np.random.randint(0, len(y_test))
    
    # Plot for each error parameter
    error_names = ['Accel Bias X', 'Accel Bias Y', 'Accel Bias Z', 
                   'Gyro Bias X', 'Gyro Bias Y', 'Gyro Bias Z']
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(y_test[seq_idx:seq_idx+100, i], 'b-', label='Actual', alpha=0.7)
        plt.plot(mlp_pred[seq_idx:seq_idx+100, i], 'r--', label='MLP Predicted', alpha=0.7)
        plt.plot(lstm_pred[seq_idx:seq_idx+100, i], 'g--', label='LSTM Predicted', alpha=0.7)
        plt.title(error_names[i])
        plt.ylabel('Error Value')
        plt.xlabel('Sample Index')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.show()
    
    # Plot error distribution
    mlp_errors = y_test - mlp_pred
    lstm_errors = y_test - lstm_pred
    
    plt.figure(figsize=(12, 8))
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.hist(mlp_errors[:, i], bins=50, alpha=0.7, label='MLP Error', density=True)
        plt.hist(lstm_errors[:, i], bins=50, alpha=0.7, label='LSTM Error', density=True)
        plt.title(f'{error_names[i]} Error Distribution')
        plt.xlabel('Error Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.show()
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print("======================")
    print(f"MLP RMSE: {mlp_rmse:.6f}")
    print(f"LSTM RMSE: {lstm_rmse:.6f}")
    print(f"Improvement: {((mlp_rmse - lstm_rmse) / mlp_rmse * 100):.2f}%")
    print(f"\nMLP MAE: {mlp_mae:.6f}")
    print(f"LSTM MAE: {lstm_mae:.6f}")
    print(f"Improvement: {((mlp_mae - lstm_mae) / mlp_mae * 100):.2f}%")

if __name__ == "__main__":
    main()