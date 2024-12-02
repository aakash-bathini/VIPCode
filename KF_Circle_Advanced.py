import numpy as np
import matplotlib.pyplot as plt

R = 10 # Radius of path 
sig = 0.5 # Standard deviation of the noise, try this at 0.5, 0.75, 1

### DONT EDIT THIS ### (Make path, measurements)
def true_path(time):
    pos_x = R * np.cos(time)
    pos_y = R * np.sin(time)
    return pos_x, pos_y

def measure(time):
    noise = np.random.normal(0,sig,2)
    pos_x = R * np.cos(time)
    pos_y = R * np.sin(time)
    return pos_x + noise[0], pos_y + noise[1]

num_pts = 50
ang_dist = 3.0/2 * np.pi
time = np.linspace(0, ang_dist, num_pts)
meas = [np.array(measure(t)) for t in time]

# This is for plotting don't use this for your filter (its perfect data)
tp = [true_path(t) for t in np.linspace(0,ang_dist,num_pts)]
### EDIT BELOW THIS ###

def kalman_filter(measurements, R, dt):
    # State: [x, y, vx, vy, ax, ay]
    # Use a constant acceleration model with circular motion constraints
    F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                  [0, 1, 0, dt, 0, 0.5*dt**2],
                  [0, 0, 1, 0, dt, 0],
                  [0, 0, 0, 1, 0, dt],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    
    # Fine-tune process noise
    Q = np.eye(6) * 0.001
    Q[4:, 4:] = np.eye(2) * 0.1  # Higher uncertainty for acceleration
    
    # Adjust measurement noise
    R_cov = np.eye(2) * (sig**2 * 1.5)
    
    # Initial state estimate
    x = np.array([measurements[0][0], measurements[0][1], 
                  -measurements[0][1]/R, measurements[0][0]/R, 0, 0])
    
    # Initial error covariance
    P = np.eye(6) * 10
    P[4:, 4:] = np.eye(2) * 100  # Higher initial uncertainty for acceleration
    
    filtered_states = []
    
    for z in measurements:
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        y = z - H @ x
        S = H @ P @ H.T + R_cov
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P
        
        # Enforce circular motion constraint
        pos = x[:2]
        vel = x[2:4]
        speed = np.linalg.norm(vel)
        tangent = np.array([-pos[1], pos[0]]) / np.linalg.norm(pos)
        x[2:4] = speed * tangent
        
        # Update acceleration to maintain circular motion
        centripetal_acc = speed**2 / R
        x[4:6] = -centripetal_acc * pos / np.linalg.norm(pos)
        
        filtered_states.append(x[:2])
    
    return np.array(filtered_states)

# Calculate dt
dt = time[1] - time[0]

# Apply Kalman filter
filtered_states = kalman_filter(meas, R, dt)

# Extract filtered x and y coordinates
data_x, data_y = filtered_states[:, 0], filtered_states[:, 1]

# Plotting (Shouldn't have to edit)
plt.figure("Position")
plt.plot(*zip(*meas), 'k.')
plt.plot(data_x, data_y, "-r")
plt.plot(*zip(*tp), 'b-')
plt.title("Position of Robot over Time")
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")
plt.grid("show")
plt.show()