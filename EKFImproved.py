import numpy as np
import matplotlib.pyplot as plt

# State vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
def process_model(x, u, dt):
    # Simplified kinematics for the drone's motion
    x_new = np.copy(x)
    x_new[0:3] += x[3:6] * dt  # Update position
    x_new[3:6] += u * dt        # Update velocity (control inputs)
    return x_new

# Measurement model
def measurement_model(x):
    # Example: GPS gives position, IMU gives attitude (Euler angles)
    z = np.copy(x[:6])  # Just taking position and velocity for the measurement
    return z

# Initialize state
state = np.zeros(9)  # Initial state vector
P = np.eye(9)        # Initial uncertainty
Q = np.eye(9) * 0.01 # Process noise covariance
R = np.eye(6) * 0.1  # Measurement noise covariance
F = np.eye(9)        # State transition matrix
B = np.eye(9)        # Control input matrix

# Synthetic sensor data generation
num_steps = 100
dt = 0.1  # Time step
control_input = np.random.randn(num_steps, 3) * 0.1  # Simulate random control inputs

# Lists to store true state, estimated state, and measurements
true_positions = []
estimated_positions = []
measurements = []

for t in range(num_steps):
    # Predict step
    state_pred = process_model(state, control_input[t], dt)
    P_pred = F @ P @ F.T + Q
    
    # Generate synthetic measurements (GPS + IMU)
    z = measurement_model(state) + np.random.randn(6) * 0.1  # Add noise
    
    # Update step (EKF)
    H = np.eye(6, 9)  # Measurement matrix
    y = z - measurement_model(state_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    state = state_pred + K @ y
    P = (np.eye(9) - K @ H) @ P_pred
    
    # Store positions for plotting
    true_positions.append(state[:3])   # True position
    estimated_positions.append(state_pred[:3])  # Estimated position
    measurements.append(z[:3])  # Measurement (GPS position)

# Convert lists to arrays for easier plotting
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)
measurements = np.array(measurements)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot true trajectory
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Position', color='b', linestyle='-', marker='o')
# Plot estimated trajectory
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Position (EKF)', color='g', linestyle='--', marker='x')
# Plot sensor measurements
plt.scatter(measurements[:, 0], measurements[:, 1], label='GPS Measurements', color='r', alpha=0.5, marker='^')

plt.title('Drone Position Estimation using EKF')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
