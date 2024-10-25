import numpy as np
from plot import plot

T = 0.1
F = np.array(
    [
        [1, 0, 0, T, 0, 0],
        [0, 1, 0, 0, T, 0],
        [0, 0, 1, 0, 0, T],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)

sigma_p = 1
sigma_v = 2
sigma_z = 3

W = T * np.array(
    [
        [sigma_p * sigma_p, 0, 0, 0, 0, 0],
        [0, sigma_p * sigma_p, 0, 0, 0, 0],
        [0, 0, sigma_p * sigma_p, 0, 0, 0],
        [0, 0, 0, sigma_v * sigma_v, 0, 0],
        [0, 0, 0, 0, sigma_v * sigma_v, 0],
        [0, 0, 0, 0, 0, sigma_v * sigma_v],
    ]
)

H = np.eye(6)

sigma_zv = 10
V = np.array(
    [
        [sigma_z * sigma_z, 0, 0, 0, 0, 0],
        [0, sigma_z * sigma_z, 0, 0, 0, 0],
        [0, 0, sigma_z * sigma_z, 0, 0, 0],
        [0, 0, 0, sigma_zv * sigma_zv, 0, 0],
        [0, 0, 0, 0, sigma_zv * sigma_zv, 0],
        [0, 0, 0, 0, 0, sigma_zv * sigma_zv],
    ]
)

x_hat = []
P = []

x_hat.append(np.array([0, 0, 0, 0, 0, 0]).T)  # Column vector
P.append(1e3 * np.eye(6))


def step(z):
    """
    Does a step with a measurement z
    """

    def prediction():
        x_hat_minus = F @ x_hat[-1]
        P_minus = F @ P[-1] @ F.T + W
        return x_hat_minus, P_minus

    def correction(z):
        x_hat_minus, P_minus = prediction()
        # Kalman gain
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + V)

        # Update
        x_hat_step = x_hat_minus + K @ (z - H @ x_hat_minus)
        P_step = (np.eye(6) - K @ H) @ P_minus

        x_hat.append(x_hat_step)
        P.append(P_step)

    correction(z)


measures = []
real_data = []
with open("filtreKalman2.txt") as f:
    for idx, line in enumerate(f):
        if idx < 5:
            continue
        elif idx < 105:
            measures.append([float(x) for x in line.split()])
            continue
        else:
            real_data.append([float(x) for x in line.split()])

real_data = np.array(real_data).reshape(6, -1)
measures = np.array(measures)

print(measures.shape)
print(real_data.shape)

for measure in measures:
    step(measure)
x_hat = np.array(x_hat).T.reshape(6, -1)
plot(real_data, T, x_hat, P, 1)
