import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class PIDController:
    def __init__(self, kp, ki, kd, dt_):
        self.error_prior = 0
        self.integral_prior = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt_

    def track(self, ref, value):
        error = ref-value
        integral = self.integral_prior + error * self.dt
        derivative = (error - self.error_prior) / self.dt
        u = self.kp * error + self.ki * integral + self.kd * derivative
        self.error_prior = error
        self.integral_prior = integral
        return min(max(u, -25), 25)    # Output saturation


class ImpedanceController:
    def __init__(self, kp, kd, dt_):
        self.ref_prior = 0
        self.value_prior = 0
        self.kp = kp
        self.kd = kd
        self.dt = dt_

    def track(self, ref, value):
        d_ref_dt = (ref - self.ref_prior) / self.dt
        d_value_dt = (value - self.value_prior) / self.dt
        u = self.kp * (ref - value) + self.kd * (d_ref_dt - d_value_dt)
        self.ref_prior = ref
        self.value_prior = value
        return min(max(u, -25), 25)  # Output saturation


class FieldController:
    pass


class NMPController:
    pass


# define seated exo model
def seated_exo(x, t_, tau_e_, tau_h_):
    # Inputs (2): tau_e = exo torque, tau_h = human torque
    # States (2): joint angle and velocity
    theta_ = x[0]
    d_theta_ = x[1]

    # Parameters
    A = 0.0001
    B = 0.0207
    J = 0.0377
    tau_g = 1.7536

    d_theta_dt = d_theta_
    dd_theta_dt = (tau_e_ + tau_h_ - B * d_theta_ - A * np.tanh(10 * d_theta_) - tau_g * np.sin(theta_)) / J

    # Return derivatives
    return [d_theta_dt, dd_theta_dt]


n_con = 2

# Time Steps
T = 4
dt = 0.002
t = np.linspace(0, T, int(T/dt) + 1)

# Initial conditions
theta0 = 0.2
d_theta0 = 0.0
y0 = [theta0, d_theta0]

# Storage for results
theta = np.ones(len(t)) * theta0
d_theta = np.ones(len(t)) * d_theta0

# Exo and Human Torques (Nm)
tau_e = np.zeros(len(t))
tau_h = np.zeros(len(t))

# Reference
theta_r = (np.cos(2 * np.pi * 0.25 * t - np.pi) + 1) / 2 + 0.2

# Controllers
pid_controller = PIDController(25, 10, 1, dt)
pid_controller.error_prior = theta_r[0] - theta[0]
imp_controller = ImpedanceController(10, 2, dt)

# Loop through each time step
for i in range(len(t) - 1):
    # Simulate
    inputs = (tau_e[i], tau_h[i])
    ts = [t[i], t[i + 1]]
    y = odeint(seated_exo, y0, ts, args=inputs)
    # Store results
    theta[i+1] = y[-1][0]
    d_theta[i+1] = y[-1][1]
    # Adjust initial condition for next loop
    y0 = y[-1]
    # Compute next control input
    tau_e[i+1] = imp_controller.track(theta_r[i], theta[i])

# Construct results and save data file
data = np.vstack((t, tau_e, tau_h, theta, d_theta))
# Transpose data
data = data.T
np.savetxt('sim_data.txt', data, delimiter=',')

# Plot the inputs
plt.figure()

plt.subplot(3, 1, 1)
plt.plot(t, tau_e, 'k-', label='tau_e')
plt.plot(t, tau_h, 'k:', label='tau_h')
plt.legend(loc='best')

plt.subplot(3, 1, 2)
plt.plot(t, theta, 'C0', label='theta')
plt.plot(t, theta_r, 'C0--', label='theta_r')
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(t, d_theta, 'C1-', label='d_theta')
plt.legend(loc='best')
plt.xlabel('Time (s)')

plt.show()
