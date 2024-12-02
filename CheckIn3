import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, LinAlgError

R = 10  
sig = 0.5  

def true_path(t):
    if t < np.pi / 2:
        pos_x = t * 2 
        pos_y = 0
    elif np.pi / 2 <= t < np.pi:
        pos_x = R * np.cos(t)
        pos_y = R * np.sin(t)
    elif np.pi <= t < 3 * np.pi / 2:
        pos_x = -R
        pos_y = (t - np.pi) * 2
    else:
        pos_x = R * np.cos(t - np.pi)
        pos_y = -R + R * np.sin(t - np.pi)
    
    noise_x = np.random.uniform(-1, 1)
    noise_y = np.random.uniform(-1, 1)
    return pos_x + noise_x, pos_y + noise_y

def measure(t):
    pos_x, pos_y = true_path(t)
    noise = np.random.normal(0, sig, 2)
    return pos_x + noise[0], pos_y + noise[1]

num_pts = 50
ang_dist = 3 * np.pi
time = np.linspace(0, ang_dist, num_pts)
measurements = [np.array(measure(t)) for t in time]
true_positions = [true_path(t) for t in time]

def ensure_positive_definite(matrix, min_eig=1e-4):
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues <= 0):
        matrix += np.eye(matrix.shape[0]) * (min_eig - np.min(eigenvalues))
    return matrix

def adjust_process_noise(Q, prev_error, current_error):
    factor = 1.1 if current_error > prev_error else 0.9
    return Q * factor

def state_transition(x, dt):
    x_pos, y_pos, x_vel, y_vel = x
    new_x_pos = x_pos + x_vel * dt
    new_y_pos = y_pos + y_vel * dt
    return np.array([new_x_pos, new_y_pos, x_vel, y_vel])

def sigma_points(x, P):
    n = len(x)
    lambda_ = 3 - n
    P = ensure_positive_definite(P)  
    sqrt_P = cholesky((n + lambda_) * P)
    
    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = x
    for i in range(n):
        sigma_pts[i + 1] = x + sqrt_P[:, i]
        sigma_pts[n + i + 1] = x - sqrt_P[:, i]
    return sigma_pts

def unscented_transform(sigma_pts, weights_mean):
    x = np.sum(weights_mean[:, None] * sigma_pts, axis=0)
    P = np.cov(sigma_pts.T, aweights=weights_mean)
    return x, P

def adaptive_ukf(z, x, P, Q, R, dt, prev_error=0):
    n = len(x)
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + 2)))
    weights_mean[0] = 2 / (n + 2)

    sigma_pts = sigma_points(x, P)
    sigma_pts_pred = np.array([state_transition(pt, dt) for pt in sigma_pts])
    
    x_pred, P_pred = unscented_transform(sigma_pts_pred, weights_mean)
    
    sigma_pts_meas = sigma_points(x_pred, P_pred)
    z_pred, S = unscented_transform(sigma_pts_meas[:, :2], weights_mean)
    
    S = ensure_positive_definite(S)

    cross_cov = np.dot(weights_mean * (sigma_pts - x_pred).T, (sigma_pts_meas[:, :2] - z_pred))
    K = np.dot(cross_cov, np.linalg.inv(S))

    x_upd = x_pred + K @ (z - z_pred)
    P_upd = P_pred - K @ S @ K.T

    current_error = np.linalg.norm(z - z_pred)
    Q = adjust_process_noise(Q, prev_error, current_error)

    return x_upd, P_upd, current_error

dt = time[1] - time[0]
Q = np.eye(4) * 0.01 + 1e-5 * np.eye(4)  
R = np.eye(2) * sig**2 + 1e-5 * np.eye(2) 
x = np.array([measurements[0][0], measurements[0][1], 0, 0])
P = np.eye(4) * 10  

filtered_states = []
prev_error = 0
for z in measurements:
    try:
        x, P, prev_error = adaptive_ukf(np.array(z), x, P, Q, R, dt, prev_error)
        filtered_states.append(x[:2])
    except LinAlgError:
        print("Matrix not positive definite, skipping this measurement.")
        filtered_states.append(filtered_states[-1] if filtered_states else np.array([0, 0]))

true_x, true_y = zip(*true_positions)
est_x, est_y = zip(*filtered_states)
error = np.sqrt(((np.array(true_x) - np.array(est_x))**2 + (np.array(true_y) - np.array(est_y))**2).mean())

plt.figure("Position")
plt.plot(*zip(*measurements), 'k.', label="Measurements")
plt.plot(est_x, est_y, "-r", label="UKF Estimated Path")
plt.plot(true_x, true_y, 'b-', label="True Path")
plt.title("UKF Tracking with True Path for Drone-like Movement")
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")
plt.legend()
plt.grid()

plt.text(0.05, 0.95, f"Mean Squared Error: {error:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
