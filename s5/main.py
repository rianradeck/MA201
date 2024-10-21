import numpy as np
import matplotlib.pyplot as plt

T = 0.1
F = np.array(
    [[1, 0, 0, T, 0, 0],
     [0, 1, 0, 0, T, 0],
     [0, 0, 1, 0, 0, T],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]]
)

sigma_p = 1
sigma_v = 2
sigma_z = 3

W = T*np.array(
    [[sigma_p*sigma_p, 0 ,0 ,0 ,0 ,0],
     [0, sigma_p*sigma_p, 0, 0, 0, 0],
     [0, 0, sigma_p*sigma_p, 0, 0, 0],
     [0, 0, 0, sigma_v*sigma_v, 0, 0],
     [0, 0, 0, 0, sigma_v*sigma_v, 0],
     [0, 0, 0, 0, 0, sigma_v*sigma_v]]
)

H = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]])

V = sigma_z*sigma_z*np.eye(3)

x_hat = []
P = []

x_hat.append(np.array([0, 0, 0, 0, 0, 0]).T) # Column vector
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
with open('filtreKalman1.txt') as f:
    for line in f:
        if '#' in line:
            temp = line.split('#')[0]
            if(len(temp) <= 100):
                continue
            real_data.append([float(x) for x in temp.split()])
            continue
        measures.append([float(x) for x in line.split()])
real_data = np.array(real_data).reshape(6, -1)
measures = np.array(measures)

print(measures.shape)
print(real_data.shape)

def plot(data, pred = None, cov = None, remove_k_first=0):
    N = data.shape[1]

    X = data[0, :]
    Y = data[1, :]
    Z = data[2, :]
    Vx = data[3, :]
    Vy = data[4, :]
    Vz = data[5, :]
    if pred is not None:
        pred = np.array(pred)
        X_pred = pred[0, remove_k_first:]
        Y_pred = pred[1, remove_k_first:]
        Z_pred = pred[2, remove_k_first:]
        Vx_pred = pred[3, remove_k_first:]
        Vy_pred = pred[4, remove_k_first:]
        Vz_pred = pred[5, remove_k_first:]
    if cov is not None:
        cov = np.array(cov)
        trace = np.trace(cov, axis1=1, axis2=2)
        sigma_x = np.sqrt(cov[remove_k_first:, 0, 0])
        sigma_y = np.sqrt(cov[remove_k_first:, 1, 1])
        sigma_z = np.sqrt(cov[remove_k_first:, 2, 2])
        sigma_vx = np.sqrt(cov[remove_k_first:, 3, 3])
        sigma_vy = np.sqrt(cov[remove_k_first:, 4, 4])
        sigma_vz = np.sqrt(cov[remove_k_first:, 5, 5])

        
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0, 0].plot(X, label='Real')
    axes[0, 0].set_title('X')

    axes[0, 1].plot(Y, label='Real')
    axes[0, 1].set_title('Y')

    axes[0, 2].plot(Z, label='Real')
    axes[0, 2].set_title('Z')

    axes[1, 0].plot(Vx, label='Real')
    axes[1, 0].set_title('Vx')

    axes[1, 1].plot(Vy, label='Real')
    axes[1, 1].set_title('Vy')

    axes[1, 2].plot(Vz, label='Real')
    axes[1, 2].set_title('Vz')

    if pred is not None:
        X_vals = list(range(remove_k_first, N+1))
        axes[0, 0].plot(X_vals, X_pred, label='Prediction')
        axes[0, 1].plot(X_vals, Y_pred, label='Prediction')
        axes[0, 2].plot(X_vals, Z_pred, label='Prediction')
        axes[1, 0].plot(X_vals, Vx_pred, label='Prediction')
        axes[1, 1].plot(X_vals, Vy_pred, label='Prediction')
        axes[1, 2].plot(X_vals, Vz_pred, label='Prediction')

    fig_, axes_ = plt.subplots(1, 1, figsize=(6, 6))
    if cov is not None:
        X_vals = list(range(remove_k_first, N+1))
        axes[0, 0].fill_between(X_vals, X_pred - sigma_x, X_pred + sigma_x, alpha=0.3, label='Confidence')
        axes[0, 1].fill_between(X_vals, Y_pred - sigma_y, Y_pred + sigma_y, alpha=0.3, label='Confidence')
        axes[0, 2].fill_between(X_vals, Z_pred - sigma_z, Z_pred + sigma_z, alpha=0.3, label='Confidence')
        axes[1, 0].fill_between(X_vals, Vx_pred - sigma_vx, Vx_pred + sigma_vx, alpha=0.3, label='Confidence')
        axes[1, 1].fill_between(X_vals, Vy_pred - sigma_vy, Vy_pred + sigma_vy, alpha=0.3, label='Confidence')
        axes[1, 2].fill_between(X_vals, Vz_pred - sigma_vz, Vz_pred + sigma_vz, alpha=0.3, label='Confidence')

        axes_.plot(trace, label='Trace of P_k')
        axes_.set_title('Trace of Cov Matrix')
        axes_.set_xlabel('k')
        axes_.set_ylabel('Trace')
        axes_.grid(alpha=0.3)
        axes_.legend()

    for i in range(2):
        for j in range(3):
            axes[i, j].legend()
            axes[i, j].grid(alpha=0.3)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

for measure in measures:
    step(measure)
x_hat = np.array(x_hat).T.reshape(6, -1)
plot(real_data, x_hat, P, 0)