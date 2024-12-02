import numpy as np
import matplotlib.pyplot as plt

# System dynamics
def f(x, dt=1.0):
    # Simple linear model
    return x

# Measurement model
def h(x):
    return x

# Prediction and Update for EKF
def ekf_predict(x, P, Q, dt=1.0):
    # Predict next state
    x_pred = f(x, dt)
    
    # Jacobian of the system dynamics (here it's an identity matrix in 1D)
    F = np.array([[1]])
    
    # Predict next covariance
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, R):
    # Compute measurement residual
    y = z - h(x_pred)
    
    # Jacobian of measurement function (identity matrix in 1D)
    H = np.array([[1]])
    
    # Measurement covariance
    S = H @ P_pred @ H.T + R
    
    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state estimate
    x_update = x_pred + K @ y
    
    # Update covariance estimate
    P_update = (np.eye(len(P_pred)) - K @ H) @ P_pred
    
    return x_update, P_update

# Initial state and covariance
x = np.array([[0]])  # initial state (e.g., position)
P = np.array([[1]])  # initial covariance estimate

# Process and measurement noise covariance
Q = np.array([[0.1]])  # process noise covariance
R = np.array([[0.5]])  # measurement noise covariance

# Store results for plotting
x_estimates = []
measurements = []
true_states = []

# Generate simulated true states and measurements
np.random.seed(0)
num_steps = 50
true_state = 0
for i in range(num_steps):
    # Simulate true state with random noise (e.g., moving object)
    true_state += np.random.normal(0, np.sqrt(Q[0, 0]))
    
    # Simulate measurement (add noise)
    z = true_state + np.random.normal(0, np.sqrt(R[0, 0]))
    
    # EKF prediction
    x_pred, P_pred = ekf_predict(x, P, Q)
    
    # EKF update
    x, P = ekf_update(x_pred, P_pred, z, R)
    
    # Store results for plotting
    true_states.append(true_state)
    measurements.append(z)
    x_estimates.append(x[0, 0])

# Plot results
plt.figure(figsize=(10, 5))

# Plot true state, measurements, and EKF estimates
plt.plot(true_states, label='True State')
plt.plot(measurements, 'o', label='Measurements', alpha=0.5)
plt.plot(x_estimates, label='EKF Estimate', linestyle='dashed')

# Calculate error
error = np.abs(np.array(true_states) - np.array(x_estimates))

plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Extended Kalman Filter Performance')
plt.grid()

# Error plot
plt.figure(figsize=(10, 2))
plt.plot(error, label='Estimation Error', color='red')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid()

plt.show()
