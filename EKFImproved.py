import numpy as np
import matplotlib.pyplot as plt

def process_model(x, u, dt):

    x_new = np.copy(x)
    x_new[0:3] += x[3:6] * dt  
    x_new[3:6] += u * dt        
    return x_new

# Measurement model
def measurement_model(x):
    z = np.copy(x[:6]) 
    return z

state = np.zeros(9)  
P = np.eye(9)       
Q = np.eye(9) * 0.01 
R = np.eye(6) * 0.1 
F = np.eye(9)        
B = np.eye(9)        

num_steps = 100
dt = 0.1  
control_input = np.random.randn(num_steps, 3) * 0.1  

true_positions = []
estimated_positions = []
measurements = []

for t in range(num_steps):
    state_pred = process_model(state, control_input[t], dt)
    P_pred = F @ P @ F.T + Q
    
    z = measurement_model(state) + np.random.randn(6) * 0.1 
    
    H = np.eye(6, 9) 
    y = z - measurement_model(state_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    state = state_pred + K @ y
    P = (np.eye(9) - K @ H) @ P_pred
    
    true_positions.append(state[:3])   
    estimated_positions.append(state_pred[:3])  
    measurements.append(z[:3]) 

true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)
measurements = np.array(measurements)

plt.figure(figsize=(10, 6))

plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Position', color='b', linestyle='-', marker='o')

plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Position (EKF)', color='g', linestyle='--', marker='x')

plt.scatter(measurements[:, 0], measurements[:, 1], label='GPS Measurements', color='r', alpha=0.5, marker='^')

plt.title('Drone Position Estimation using EKF')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
