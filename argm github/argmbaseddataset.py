import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

class MEMS_IMU_Calibration_Generator:
    def __init__(self, fs=200, duration=10800, temperature=20):
        """
        Initialize MEMS IMU calibration dataset generator
        
        Parameters:
        fs: Sampling frequency (Hz) - based on ADIS16364 spec (200 Hz)
        duration: Duration in seconds (3 hours as in paper)
        temperature: Temperature in °C (from -40 to +60 as in paper)
        """
        self.fs = fs
        self.duration = duration
        self.temperature = temperature
        self.n_samples = int(fs * duration)
        self.time = np.arange(0, duration, 1/fs)
        
        # Initialize sensor parameters based on ADIS16364 specifications
        self.initialize_sensor_parameters()
        
    def initialize_sensor_parameters(self):
        """Initialize sensor parameters based on ADIS16364 specifications"""
        # Gyro parameters (from Table 1)
        self.gyro_initial_bias_error = 0.03  # ±3%
        self.gyro_inrun_bias_stability = 0.00007  # 0.007%
        self.gyro_bias_temp_coeff = 0.0001  # ±0.01%/℃
        self.gyro_arw = 2 * (np.pi/180) / 60  # 2°/√h to rad/s/√Hz
        
        # Accelerometer parameters (from Table 1)
        self.accel_initial_bias_error = 0.008 * 9.81  # ±8 mg to m/s²
        self.accel_inrun_bias_stability = 0.0001 * 9.81  # 0.1 mg to m/s²
        self.accel_bias_temp_coeff = 0.00005 * 9.81  # ±0.05 mg/℃ to m/s²/℃
        self.accel_vrw = 0.12 / 60  # 0.12 m/s/√h to m/s²/√Hz
        
        # Temperature-dependent parameters (based on paper findings)
        self.calculate_temperature_dependencies()
        
    def calculate_temperature_dependencies(self):
        """Calculate temperature-dependent parameters based on paper findings"""
        # Correlation time temperature dependency (from Figs 8-9)
        # Using exponential decay model based on paper's results
        T_ref = 20  # Reference temperature from paper
        
        # Gyro correlation time model (approximated from Fig 8)
        self.gyro_tc_x = 300 * np.exp(-0.015 * abs(self.temperature - T_ref))
        self.gyro_tc_y = 280 * np.exp(-0.014 * abs(self.temperature - T_ref))
        self.gyro_tc_z = 320 * np.exp(-0.016 * abs(self.temperature - T_ref))
        
        # Accelerometer correlation time model (approximated from Fig 9)
        self.accel_tc_x = 250 * np.exp(-0.012 * abs(self.temperature - T_ref))
        self.accel_tc_y = 230 * np.exp(-0.011 * abs(self.temperature - T_ref))
        self.accel_tc_z = 270 * np.exp(-0.013 * abs(self.temperature - T_ref))
        
        # Bias temperature dependency
        self.gyro_bias_temp_effect = self.gyro_bias_temp_coeff * (self.temperature - 20)
        self.accel_bias_temp_effect = self.accel_bias_temp_coeff * (self.temperature - 20)
        
    def generate_ar_based_gm_process(self, tc, noise_variance, n_samples):
        """
        Generate AR-based Gauss-Markov process as described in the paper
        
        Parameters:
        tc: Correlation time (seconds)
        noise_variance: Variance of the driving noise
        n_samples: Number of samples to generate
        
        Returns:
        GM process samples
        """
        # Calculate AR parameter from correlation time (Equation 10)
        dt = 1/self.fs
        c1 = np.exp(-dt/tc)
        
        # Generate the AR-based GM process (Equation 3)
        process = np.zeros(n_samples)
        w = np.random.normal(0, np.sqrt(noise_variance), n_samples)
        
        for k in range(1, n_samples):
            process[k] = c1 * process[k-1] + w[k]
            
        return process
    
    def generate_sensor_data(self):
        """Generate complete MEMS IMU calibration dataset"""
        print(f"Generating MEMS IMU calibration data at {self.temperature}°C...")
        
        # Generate ground truth (stationary IMU)
        true_accel = np.zeros((self.n_samples, 3))
        true_gyro = np.zeros((self.n_samples, 3))
        
        # Gravity vector (IMU stationary on level surface)
        true_accel[:, 2] = 9.81  # z-axis points up
        
        # Generate sensor errors
        sensor_errors = self.generate_sensor_errors()
        
        # Create raw sensor outputs (true values + errors)
        raw_accel = true_accel + sensor_errors['accel_bias'] + sensor_errors['accel_noise']
        raw_gyro = true_gyro + sensor_errors['gyro_bias'] + sensor_errors['gyro_noise']
        
        # Create dataset
        dataset = {
            'timestamp': self.time,
            'temperature': np.full(self.n_samples, self.temperature),
            'raw_accel_x': raw_accel[:, 0],
            'raw_accel_y': raw_accel[:, 1],
            'raw_accel_z': raw_accel[:, 2],
            'raw_gyro_x': raw_gyro[:, 0],
            'raw_gyro_y': raw_gyro[:, 1],
            'raw_gyro_z': raw_gyro[:, 2],
            'true_accel_x': true_accel[:, 0],
            'true_accel_y': true_accel[:, 1],
            'true_accel_z': true_accel[:, 2],
            'true_gyro_x': true_gyro[:, 0],
            'true_gyro_y': true_gyro[:, 1],
            'true_gyro_z': true_gyro[:, 2],
            'accel_bias_x': sensor_errors['accel_bias'][:, 0],
            'accel_bias_y': sensor_errors['accel_bias'][:, 1],
            'accel_bias_z': sensor_errors['accel_bias'][:, 2],
            'gyro_bias_x': sensor_errors['gyro_bias'][:, 0],
            'gyro_bias_y': sensor_errors['gyro_bias'][:, 1],
            'gyro_bias_z': sensor_errors['gyro_bias'][:, 2],
            'accel_scale_x': sensor_errors['accel_scale'][:, 0],
            'accel_scale_y': sensor_errors['accel_scale'][:, 1],
            'accel_scale_z': sensor_errors['accel_scale'][:, 2],
            'gyro_scale_x': sensor_errors['gyro_scale'][:, 0],
            'gyro_scale_y': sensor_errors['gyro_scale'][:, 1],
            'gyro_scale_z': sensor_errors['gyro_scale'][:, 2],
        }
        
        return pd.DataFrame(dataset)
    
    def generate_sensor_errors(self):
        """Generate all sensor error components"""
        errors = {}
        
        # Generate bias errors using AR-based GM process
        errors['accel_bias'] = self.generate_bias_errors('accelerometer')
        errors['gyro_bias'] = self.generate_bias_errors('gyro')
        
        # Generate scale factor errors (slowly varying)
        errors['accel_scale'] = self.generate_scale_errors('accelerometer')
        errors['gyro_scale'] = self.generate_scale_errors('gyro')
        
        # Generate white noise components
        errors['accel_noise'] = self.generate_white_noise('accelerometer')
        errors['gyro_noise'] = self.generate_white_noise('gyro')
        
        return errors
    
    def generate_bias_errors(self, sensor_type):
        """Generate bias errors using AR-based GM process"""
        bias = np.zeros((self.n_samples, 3))
        
        if sensor_type == 'accelerometer':
            # Initial bias with temperature effect
            initial_bias = self.accel_initial_bias_error + self.accel_bias_temp_effect
            noise_variance = self.accel_inrun_bias_stability**2
            
            # Generate GM processes for each axis
            bias[:, 0] = initial_bias + self.generate_ar_based_gm_process(
                self.accel_tc_x, noise_variance, self.n_samples)
            bias[:, 1] = initial_bias + self.generate_ar_based_gm_process(
                self.accel_tc_y, noise_variance, self.n_samples)
            bias[:, 2] = initial_bias + self.generate_ar_based_gm_process(
                self.accel_tc_z, noise_variance, self.n_samples)
                
        else:  # gyro
            initial_bias = self.gyro_initial_bias_error + self.gyro_bias_temp_effect
            noise_variance = self.gyro_inrun_bias_stability**2
            
            bias[:, 0] = initial_bias + self.generate_ar_based_gm_process(
                self.gyro_tc_x, noise_variance, self.n_samples)
            bias[:, 1] = initial_bias + self.generate_ar_based_gm_process(
                self.gyro_tc_y, noise_variance, self.n_samples)
            bias[:, 2] = initial_bias + self.generate_ar_based_gm_process(
                self.gyro_tc_z, noise_variance, self.n_samples)
                
        return bias
    
    def generate_scale_errors(self, sensor_type):
        """Generate scale factor errors (slowly varying)"""
        scale = np.zeros((self.n_samples, 3))
        
        if sensor_type == 'accelerometer':
            # Scale factor error ~0.1% with slow variation
            base_error = 0.001
            # Add slow variation using low-pass filtered noise
            white_noise = np.random.normal(0, 0.0002, self.n_samples)
            b, a = signal.butter(2, 0.0001, 'lowpass')
            slow_variation = signal.filtfilt(b, a, white_noise)
            
            for i in range(3):
                scale[:, i] = base_error + slow_variation
                
        else:  # gyro
            base_error = 0.002  # Gyro typically has larger scale errors
            white_noise = np.random.normal(0, 0.0003, self.n_samples)
            b, a = signal.butter(2, 0.00005, 'lowpass')  # Even slower variation
            slow_variation = signal.filtfilt(b, a, white_noise)
            
            for i in range(3):
                scale[:, i] = base_error + slow_variation
                
        return scale
    
    def generate_white_noise(self, sensor_type):
        """Generate white noise component"""
        noise = np.zeros((self.n_samples, 3))
        
        if sensor_type == 'accelerometer':
            std_dev = self.accel_vrw * np.sqrt(self.fs)
            for i in range(3):
                noise[:, i] = np.random.normal(0, std_dev, self.n_samples)
        else:  # gyro
            std_dev = self.gyro_arw * np.sqrt(self.fs)
            for i in range(3):
                noise[:, i] = np.random.normal(0, std_dev, self.n_samples)
                
        return noise

# Example usage and validation
if __name__ == "__main__":
    # Generate dataset at different temperatures as in the paper
    temperatures = [-40, -20, 0, 20, 40, 60]
    datasets = {}
    
    for temp in temperatures:
        generator = MEMS_IMU_Calibration_Generator(temperature=temp)
        dataset = generator.generate_sensor_data()
        datasets[temp] = dataset
        
        # Save to CSV
        dataset.to_csv(f'mems_imu_calibration_{temp}C.csv', index=False)
        print(f"Dataset for {temp}°C saved with {len(dataset)} samples")
    
    # Analyze and plot some results
    plt.figure(figsize=(12, 8))
    
    # Plot bias behavior at different temperatures
    for i, temp in enumerate(temperatures):
        dataset = datasets[temp]
        # Plot first 100 seconds of Z-axis accelerometer bias
        plt.subplot(2, 1, 1)
        plt.plot(dataset['timestamp'][:2000], dataset['accel_bias_z'][:2000], 
                label=f'{temp}°C')
        
        plt.subplot(2, 1, 2)
        plt.plot(dataset['timestamp'][:2000], dataset['gyro_bias_z'][:2000], 
                label=f'{temp}°C')
    
    plt.subplot(2, 1, 1)
    plt.title('Accelerometer Z-axis Bias at Different Temperatures (First 100s)')
    plt.ylabel('Bias (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title('Gyro Z-axis Bias at Different Temperatures (First 100s)')
    plt.ylabel('Bias (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('temperature_dependent_bias.png')
    plt.show()
    
    print("Dataset generation completed!")