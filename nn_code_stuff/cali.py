# Script to apply INS calibration equations
# Loads 4 prediction CSVs (bias, sfp, sfn, msal) and test CSV (raw accel/temp)
# Applies equations: y_x = bias_x + sf_x*a_x + m_xy*a_y + m_xz*a_z, etc.
# Uses sfp if temp >= 0, sfn if temp < 0 for each axis independently
# Saves calibrated accel, raw accel, and differences as calibrated_accel.csv
# Plots raw vs calibrated for each axis (x, y, z)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User inputs: Replace with your file paths if needed
TEST_CSV = r'C:\\THESIS_PINNS\\temp_storage\\nn_code_stuff\\test.csv'  # Test CSV with raw accel_x,y,z temp_x,y,z (columns 0-5)
BIAS_PRED_CSV = r'C:\\THESIS_PINNS\\temp_storage\\nn_code_stuff\\bias_preds.csv'  # From bias run: columns ['Bias X', 'Bias Y', 'Bias Z']
SFP_PRED_CSV = r'C:\\THESIS_PINNS\\temp_storage\\nn_code_stuff\\sfp_preds.csv'    # From sfp run: columns ['SFP X', 'SFP Y', 'SFP Z']
SFN_PRED_CSV = r'C:\\THESIS_PINNS\\temp_storage\\nn_code_stuff\\sfn_preds.csv'    # From sfn run: columns ['SFN X', 'SFN Y', 'SFN Z']
MSAL_PRED_CSV = r'C:\\THESIS_PINNS\\temp_storage\\nn_code_stuff\\msal_preds.csv'  # From msal run: columns ['M XY', 'M XZ', 'M YX', 'M YZ', 'M ZX', 'M ZY']

# Load test CSV for raw values
test_df = pd.read_csv(TEST_CSV)
num_samples = len(test_df)
print(f"Loaded {num_samples} test samples for calibration.")

# Load prediction CSVs
try:
    bias_df = pd.read_csv(BIAS_PRED_CSV)
    sfp_df = pd.read_csv(SFP_PRED_CSV)
    sfn_df = pd.read_csv(SFN_PRED_CSV)
    msal_df = pd.read_csv(MSAL_PRED_CSV)
except FileNotFoundError as e:
    print(f"Error: Prediction file not found: {e}")
    exit()

# Verify number of rows match test samples
if not all(len(df) == num_samples for df in [bias_df, sfp_df, sfn_df, msal_df]):
    print("Error: All prediction CSVs must have the same number of rows as test CSV.")
    exit()

# Verify prediction shapes
if (bias_df.shape[1] != 3 or sfp_df.shape[1] != 3 or sfn_df.shape[1] != 3 or msal_df.shape[1] != 6):
    print("Error: Prediction CSVs must have correct number of columns (bias/sfp/sfn: 3, msal: 6).")
    exit()

# Extract predictions (assume column order: x, y, z for bias/sfp/sfn; xy,xz,yx,yz,zx,zy for msal)
bias = bias_df.values  # [samples, 3]
sfp = sfp_df.values    # [samples, 3]
sfn = sfn_df.values    # [samples, 3]
msal = msal_df.values  # [samples, 6]

# Extract raw accel and temp from test_df (assume columns 0-2: accel_x,y,z; 3-5: temp_x,y,z)
raw_a = test_df.iloc[:, 0:3].values  # [samples, 3]: a_x, a_y, a_z
raw_t = test_df.iloc[:, 3:6].values  # [samples, 3]: temp_x, temp_y, temp_z

# Compute calibrated accel for each sample
calib_accel = np.zeros((num_samples, 3))  # [samples, 3]: y_x, y_y, y_z

for j in range(num_samples):
    a_x, a_y, a_z = raw_a[j, 0], raw_a[j, 1], raw_a[j, 2]
    t_x, t_y, t_z = raw_t[j, 0], raw_t[j, 1], raw_t[j, 2]
    
    bias_x, bias_y, bias_z = bias[j, 0], bias[j, 1], bias[j, 2]
    sfp_x, sfp_y, sfp_z = sfp[j, 0], sfp[j, 1], sfp[j, 2]
    sfn_x, sfn_y, sfn_z = sfn[j, 0], sfn[j, 1], sfn[j, 2]
    m_xy, m_xz, m_yx, m_yz, m_zx, m_zy = msal[j, 0], msal[j, 1], msal[j, 2], msal[j, 3], msal[j, 4], msal[j, 5]
    
    # Choose sf based on temp for each axis
    sf_x = sfn_x if t_x < 0 else sfp_x
    sf_y = sfn_y if t_y < 0 else sfp_y
    sf_z = sfn_z if t_z < 0 else sfp_z
    
    # Calibrated values
    y_x = bias_x + sf_x * a_x + m_xy * a_y + m_xz * a_z
    y_y = bias_y + sf_y * a_y + m_yx * a_x + m_yz * a_z
    y_z = bias_z + sf_z * a_z + m_zx * a_x + m_zy * a_y
    
    calib_accel[j] = [y_x, y_y, y_z]

# Compute differences: raw - calibrated
diff_accel = raw_a - calib_accel  # [samples, 3]: diff_x, diff_y, diff_z

# Save raw, calibrated, and differences to CSV
output_df = pd.DataFrame({
    'raw_x': raw_a[:, 0],
    'raw_y': raw_a[:, 1],
    'raw_z': raw_a[:, 2],
    'calib_x': calib_accel[:, 0],
    'calib_y': calib_accel[:, 1],
    'calib_z': calib_accel[:, 2],
    'diff_x': diff_accel[:, 0],
    'diff_y': diff_accel[:, 1],
    'diff_z': diff_accel[:, 2]
})
output_df.to_csv('calibrated_accel.csv', index=False)
print("Saved raw, calibrated, and differences to calibrated_accel.csv")

# Plot raw vs calibrated for each axis
for axis, label in enumerate(['X', 'Y', 'Z']):
    plt.figure(figsize=(6, 4))
    plt.plot(raw_a[:, axis], label='Raw', marker='o')
    plt.plot(calib_accel[:, axis], label='Calibrated', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Accel {label} Raw vs Calibrated')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'calib_{label}_plot.png')
    plt.show()

print("Calibration and plotting complete.")