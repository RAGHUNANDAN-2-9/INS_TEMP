# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for neat, publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_csvs(csv_paths, is_train=True):
    """
    Load one or multiple CSV files and merge them into a single DataFrame.
    Assumes columns: first 6 are inputs (accel_x, accel_y, accel_z, temp_x, temp_y, temp_z),
    next 15 are targets (error params like bias, sf, misalignment).
    
    Args:
        csv_paths (list): List of paths to CSV files.
        is_train (bool): If True, expect 150 samples; for test, expect 14 samples, 21 columns.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if not isinstance(csv_paths, list):
        csv_paths = [csv_paths]
    
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Ensure exactly 21 columns
        if df.shape[1] != 21:
            raise ValueError(f"CSV at {path} must have exactly 21 columns (6 inputs + 15 targets). Found {df.shape[1]}.")
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    expected_samples = 150 if is_train else 14
    print(f"Loaded {len(merged_df)} samples from {'train' if is_train else 'test'} CSV(s). Expected ~{expected_samples}.")
    return merged_df

def prepare_data(df, standardize=True, target_cols=None, val_split=0.2):
    """
    Prepare data: split inputs/targets, standardize if requested, split train/val if training.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        standardize (bool): Whether to apply StandardScaler (False if errors are heavy).
        target_cols (list): List of target column indices or names; if None, use last 15 columns.
        val_split (float): Fraction for validation split (only for training data).
    
    Returns:
        For training: X_train, X_val, y_train, y_val (tensors), scalers
        For testing: X_test, y_test (tensors), scalers
    """
    # Assume first 6 columns are inputs, rest targets
    input_cols = df.columns[:6]
    if target_cols is None:
        target_cols = df.columns[6:]
    else:
        target_cols = [col if isinstance(col, str) else df.columns[col] for col in target_cols]
    
    X = df[input_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    
    if standardize:
        # Standardize inputs and targets separately (common for INS calibration)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
        print("Applied standardization. Scalers saved for inverse transform.")
    else:
        scaler_X = scaler_y = None
        print("Skipped standardization due to reported heavy errors.")
    
    if val_split > 0:  # Training mode: split train/val
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        return train_dataset, val_dataset, scaler_X, scaler_y
    else:  # Testing mode: return full
        test_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return test_dataset, None, scaler_X, scaler_y

class INSNet(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for INS calibration.
    Input: 6 features (accel/temp XYZ).
    Output: 15 targets (error params); flexible for fewer via output_dim.
    At least 3 hidden layers; easy to tweak via hidden_dims.
    """
    def __init__(self, input_dim=6, output_dim=15, hidden_dims=[64, 32, 16], dropout=0.2):
        super(INSNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers (tweak hidden_dims for different architectures)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, num_epochs=200, lr=0.001, patience=20):
    """
    Train the model with early stopping and LR scheduler.
    Plots training/validation losses.
    
    Args:
        model: PyTorch model.
        train_loader, val_loader: DataLoaders.
        num_epochs (int): Max epochs.
        lr (float): Initial learning rate.
        patience (int): Early stopping patience.
    
    Returns:
        history (dict): Losses per epoch.
        model: Trained model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_losses.png', bbox_inches='tight')
    plt.show()
    
    return history, model

def evaluate_model(model, test_loader, num_targets=15, scaler_y=None):
    """
    Evaluate on test set: predictions, metrics, plots.
    Handles small error values with tight axes and no overlap in scatter plots.
    
    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        num_targets (int): Number of targets (for subplots).
        scaler_y: StandardScaler for targets (if standardized).
    
    Returns:
        metrics (dict): MSE, MAE per target and overall.
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    y_pred = np.vstack(predictions)
    y_true = np.vstack(actuals)
    
    if scaler_y is not None:
        # Inverse transform predictions and actuals for metrics/plots
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_true)
        print("Inverse transformed predictions and actuals for evaluation.")
    
    # Metrics
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'per_target_mse': mse,
        'per_target_mae': mae
    }
    
    print(f"Overall Test MSE: {overall_mse:.8f}, MAE: {overall_mae:.8f}")
    for i, (m, a) in enumerate(zip(mse, mae)):
        print(f"Target {i+1}: MSE={m:.8f}, MAE={a:.8f}")
    
    # Plots: Actual vs Predicted (subplots for each target to avoid overlap)
    fig, axes = plt.subplots((num_targets + 4) // 5, min(5, num_targets), figsize=(min(5, num_targets) * 3, ((num_targets + 4) // 5) * 3))
    axes = np.array(axes).ravel() if num_targets > 1 else [axes]
    
    for i in range(num_targets):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, s=30, color='blue')
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        # Tight limits with small padding for small values
        padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-6 else 1e-6
        ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', lw=2)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Target {i+1} (Error Param {i+1})')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted (Test Set)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('actual_vs_predicted.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Residuals plot for error analysis
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    for i in range(min(5, num_targets)):
        plt.subplot(2, 3, i+1)
        plt.scatter(y_pred[:, i], residuals[:, i], alpha=0.7, s=30)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.title(f'Residuals Target {i+1}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residuals.png', bbox_inches='tight')
    plt.show()
    
    return metrics

def main(train_csv_paths, test_csv_path, standardize=False, target_indices=None, batch_size=32, num_epochs=200, lr=0.001, model_configs=None):
    """
    Main function: Load data, train, evaluate.
    Supports multiple models via model_configs list.
    Each config is a dict with 'hidden_dims' and 'dropout'.
    User provides CSV paths as lists.
    
    Args:
        train_csv_paths (list): List of train CSV paths.
        test_csv_path (list or str): Test CSV path(s).
        standardize (bool): Apply standardization (default False due to reported errors).
        target_indices (list): Indices of target columns (0-based from target start, e.g., [0,1,2] for first 3).
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        model_configs (list): List of dicts [{'hidden_dims': [64,32,16], 'dropout': 0.2}, ...].
    
    Returns:
        None; saves models, metrics, plots.
    """
    # Default model configs (try different architectures)
    if model_configs is None:
        model_configs = [
            {'hidden_dims': [64, 32, 16], 'dropout': 0.2},  # Base model
            {'hidden_dims': [128, 64, 32], 'dropout': 0.3},  # Deeper
            {'hidden_dims': [256, 128, 64, 32], 'dropout': 0.4}  # Wider + deeper
        ]
    
    # Load data
    train_df = load_csvs(train_csv_paths, is_train=True)
    test_df = load_csvs(test_csv_path, is_train=False)
    
    # Prepare data (target_indices adjusted to full column indices if provided)
    num_targets = len(target_indices) if target_indices is not None else 15
    target_cols = [6 + idx for idx in target_indices] if target_indices is not None else None
    
    train_dataset, val_dataset, scaler_X, scaler_y = prepare_data(train_df, standardize=standardize, target_cols=target_cols, val_split=0.2)
    test_dataset, _, _, test_scaler_y = prepare_data(test_df, standardize=standardize, target_cols=target_cols, val_split=0)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Train and evaluate each model
    for idx, config in enumerate(model_configs):
        print(f"\nTraining Model {idx+1} with config: {config}")
        model = INSNet(
            input_dim=6,
            output_dim=num_targets,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(device)
        print(model)
        
        # Train
        history, trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)
        
        # Evaluate
        metrics = evaluate_model(trained_model, test_loader, num_targets=num_targets, scaler_y=test_scaler_y)
        
        # Save model and metrics
        torch.save(trained_model.state_dict(), f'trained_ins_model_{idx+1}.pth')
        pd.DataFrame(metrics).to_csv(f'test_metrics_model_{idx+1}.csv')
        print(f"Model {idx+1} saved as 'trained_ins_model_{idx+1}.pth'.")

# Example usage
if __name__ == "__main__":
    # User provides lists of CSV paths here
    TRAIN_CSV_PATHS = ['path/to/train1.csv', 'path/to/train2.csv']  # Replace with your paths
    TEST_CSV_PATHS = ['path/to/test.csv']  # Replace with your test path
    
    # Example with 3 targets (bias parameters); set to None for all 15
    TARGET_INDICES = [0, 1, 2]  # First 3 targets; set to None for all 15
    
    # Run with default configs or modify model_configs
    main(
        train_csv_paths=TRAIN_CSV_PATHS,
        test_csv_path=TEST_CSV_PATHS,
        standardize=False,  # Set to True if standardization works better
        target_indices=TARGET_INDICES,
        batch_size=32,
        num_epochs=200,
        lr=0.001,
        model_configs=[
            {'hidden_dims': [64, 32, 16], 'dropout': 0.2},
            {'hidden_dims': [128, 64, 32], 'dropout': 0.3},
            {'hidden_dims': [256, 128, 64, 32], 'dropout': 0.4}
        ]
    )