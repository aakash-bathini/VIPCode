import numpy as np
from casadi import *
import matplotlib.pyplot as plt

class DronePositionEstimator:
    def __init__(self):
        """
        EKF-based position estimator for a drone
        State vector: [x, y, z, vx, vy, vz, phi, theta, psi]
            x,y,z: position
            vx,vy,vz: velocities
            phi,theta,psi: roll, pitch, yaw angles
        Measurements:
            - GPS (x, y, z)
            - IMU (accelerations ax, ay, az, angular rates p, q, r)
            - Barometer (z)
            - Magnetometer (heading)
        """
        # State dimension: position(3) + velocity(3) + attitude(3)
        self.nx = 9
        
        # Measurement dimension: GPS(3) + IMU(6) + Baro(1) + Mag(1)
        self.nz = 11
        
        # Create symbolic variables
        self.x = SX.sym('x', self.nx)  # State
        self.u = SX.sym('u', 6)        # Control input (desired thrust and angular rates)
        self.z = SX.sym('z', self.nz)  # Measurements
        
        # Extract state variables for readability
        self.pos = self.x[0:3]    # Position
        self.vel = self.x[3:6]    # Velocity
        self.att = self.x[6:9]    # Attitude (roll, pitch, yaw)
        
        # Physical parameters
        self.g = 9.81             # Gravity
        self.mass = 1.5           # Drone mass in kg
        self.dt = 0.01            # Time step in seconds
        
        # Initialize noise covariances
        self._initialize_covariances()
        
        # Set up process and measurement models
        self._setup_process_model()
        self._setup_measurement_model()
        
    def _initialize_covariances(self):
        """Initialize process and measurement noise covariances"""
        # Process noise
        self.Q = np.diag([
            0.1, 0.1, 0.1,        # Position noise
            0.2, 0.2, 0.2,        # Velocity noise
            0.1, 0.1, 0.1         # Attitude noise
        ])
        
        # Measurement noise
        self.R = np.diag([
            2.0, 2.0, 3.0,        # GPS position noise (x,y,z)
            0.1, 0.1, 0.1,        # IMU acceleration noise
            0.01, 0.01, 0.01,     # IMU angular rate noise
            0.5,                   # Barometer noise
            0.1                    # Magnetometer noise
        ])
        
    def _rotation_matrix(self, phi, theta, psi):
        """Create rotation matrix from euler angles"""
        # Define rotation matrices for each axis
        R_x = vertcat(
            horzcat(1, 0, 0),
            horzcat(0, cos(phi), -sin(phi)),
            horzcat(0, sin(phi), cos(phi))
        )
        
        R_y = vertcat(
            horzcat(cos(theta), 0, sin(theta)),
            horzcat(0, 1, 0),
            horzcat(-sin(theta), 0, cos(theta))
        )
        
        R_z = vertcat(
            horzcat(cos(psi), -sin(psi), 0),
            horzcat(sin(psi), cos(psi), 0),
            horzcat(0, 0, 1)
        )
        
        return R_z @ R_y @ R_x
    
    def _setup_process_model(self):
        """Set up the nonlinear process model"""
        # Extract euler angles
        phi, theta, psi = self.att[0], self.att[1], self.att[2]
        
        # Get rotation matrix
        R = self._rotation_matrix(phi, theta, psi)
        
        # Extract control inputs
        thrust = self.u[0]
        omega = self.u[1:4]  # Angular rates
        
        # Acceleration in body frame
        a_body = vertcat(0, 0, thrust/self.mass)
        
        # Transform to inertial frame and add gravity
        a_inertial = R @ a_body + vertcat(0, 0, -self.g)
        
        # State derivatives
        pos_dot = self.vel
        vel_dot = a_inertial
        
        # Attitude kinematics (simplified)
        att_dot = omega
        
        # Complete state derivative
        x_dot = vertcat(pos_dot, vel_dot, att_dot)
        
        # Discrete time update using RK4
        k1 = x_dot
        k2 = substitute(x_dot, self.x, self.x + self.dt/2 * k1)
        k3 = substitute(x_dot, self.x, self.x + self.dt/2 * k2)
        k4 = substitute(x_dot, self.x, self.x + self.dt * k3)
        
        f = self.x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Create function objects
        self.f_func = Function('f', [self.x, self.u], [f])
        self.F_func = Function('F', [self.x, self.u], [jacobian(f, self.x)])
        
    def _setup_measurement_model(self):
        """Set up the nonlinear measurement model"""
        # GPS measurement (position)
        h_gps = self.pos
        
        # IMU measurements (accelerations and angular rates)
        phi, theta, psi = self.att[0], self.att[1], self.att[2]
        R = self._rotation_matrix(phi, theta, psi)
        
        # Specific force measurement (including gravity)
        f_b = transpose(R) @ (self.vel - vertcat(0, 0, -self.g))
        
        # Angular rates (directly from state derivatives)
        omega = self.att  # Simplified - in reality would come from state derivatives
        
        # Barometer (altitude)
        h_baro = self.pos[2]
        
        # Magnetometer (heading)
        h_mag = self.att[2]  # Simplified - in reality would include magnetic declination
        
        # Complete measurement model
        h = vertcat(h_gps, f_b, omega, h_baro, h_mag)
        
        # Create function objects
        self.h_func = Function('h', [self.x], [h])
        self.H_func = Function('H', [self.x], [jacobian(h, self.x)])
        
    def predict(self, x_prev, P_prev, u):
        """
        Prediction step
        
        Args:
            x_prev: Previous state estimate (9x1)
            P_prev: Previous covariance matrix (9x9)
            u: Control input (6x1)
        """
        # Predict state
        x_pred = np.array(self.f_func(x_prev, u)).flatten()
        
        # Calculate Jacobian
        F = np.array(self.F_func(x_prev, u))
        
        # Predict covariance
        P_pred = F @ P_prev @ F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, z_meas):
        """
        Update step
        
        Args:
            x_pred: Predicted state (9x1)
            P_pred: Predicted covariance (9x9)
            z_meas: Measurement vector (11x1)
        """
        # Calculate expected measurement
        z_pred = np.array(self.h_func(x_pred)).flatten()
        
        # Calculate measurement Jacobian
        H = np.array(self.H_func(x_pred))
        
        # Innovation and innovation covariance
        y = z_meas - z_pred
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        x_post = x_pred + K @ y
        P_post = (np.eye(self.nx) - K @ H) @ P_pred
        
        return x_post, P_post
    
    def normalize_angles(self, x):
        """Normalize euler angles to [-pi, pi]"""
        x[6:9] = np.mod(x[6:9] + np.pi, 2 * np.pi) - np.pi
        return x

# Example usage
def run_drone_estimator():
    """Example of running the drone position estimator"""
    # Initialize estimator
    estimator = DronePositionEstimator()
    
    # Initial state and covariance
    x = np.zeros(9)  # All states initialized to zero
    P = np.eye(9) * 0.1  # Initial uncertainty
    
    # Simulated measurements (example values)
    z = np.array([
        0.1, 0.2, 0.3,    # GPS position
        0.0, 0.0, 9.81,   # IMU accelerations
        0.0, 0.0, 0.0,    # IMU angular rates
        0.3,              # Barometer
        0.0               # Magnetometer
    ])
    
    # Control inputs (example values)
    u = np.array([
        9.81 * 1.5,       # Thrust (hovering)
        0.0, 0.0, 0.0,    # Desired angular rates
        0.0, 0.0          # Additional control inputs if needed
    ])
    
    # Run one step of the estimator
    x_pred, P_pred = estimator.predict(x, P, u)
    x_post, P_post = estimator.update(x_pred, P_pred, z)
    
    # Normalize angles
    x_post = estimator.normalize_angles(x_post)
    
    return x_post, P_post

    # Simulate data from the drone position estimator
def simulate_drone_position():
    timesteps = 100
    dt = 0.1  # Time step
    time = np.linspace(0, timesteps * dt, timesteps)

    # Simulated position data (example: sine and cosine motion)
    x = np.sin(time)  # Position along x-axis
    y = np.cos(time)  # Position along y-axis
    z = 0.1 * time    # Position along z-axis (linear ascent)

    return time, x, y, z

# Generate simulated data
time, x, y, z = simulate_drone_position()

# Plot the positions
plt.figure(figsize=(10, 6))
plt.plot(time, x, label='x (Position along x-axis)', color='red')
plt.plot(time, y, label='y (Position along y-axis)', color='blue')
plt.plot(time, z, label='z (Position along z-axis)', color='green')
plt.title('Simulated Drone Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()
