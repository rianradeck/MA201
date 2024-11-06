import numpy as np
import matplotlib.pyplot as plt


# Paramètres donnés
T = 0.01
sigma_v = 0.1
sigma_omega = 0.01
sigma_w = 0.2
V = 3.0  # m/s
omega = 2 * np.pi / 3  # rad/s
x_b, y_b = 2, 5  # Coordonnées du point fixe
x0 = np.array([[0, 0, 0]]).T  # État initial Estimated
P0 = 10 * np.eye(3)  # Covariance initiale

# Matrice de covariance du bruit de processus (pour u = [ΔV, Δω]^T)
Cov_u = np.diag([sigma_v**2, sigma_omega**2])

# Matrice de covariance du bruit de mesure
W_k = sigma_w**2

def compute_Phi_k(V, T, phi_hat):
    return np.array([
        [1, 0, -V * T * np.sin(phi_hat)],
        [0, 1,  V * T * np.cos(phi_hat)],
        [0, 0, 1]
    ])

def compute_Gamma(T, phi_hat):
    return np.array([
        [T * np.cos(phi_hat), 0],
        [T * np.sin(phi_hat), 0],
        [0, T]
    ])


def compute_H_k(x_hat, y_hat):
    distance = np.sqrt((x_hat - x_b)**2 + (y_hat - y_b)**2)
    return np.array([
        [(x_hat - x_b) / distance],
        [(y_hat - y_b) / distance],
        [0]
    ]).T

X_hat = [x0]
P = [P0]

def step(z):
    def prediction():
        x_hat_k, y_hat_k, omega_hat_k = X_hat[-1][0][0], X_hat[-1][1][0], X_hat[-1][2][0]
        X_hat_pred = np.array([
            [x_hat_k + V * T * np.cos(omega_hat_k)],
            [y_hat_k + V * T * np.sin(omega_hat_k)],
            [omega_hat_k + T * omega]
        ])
        Phi_k = compute_Phi_k(V, T, omega_hat_k)
        Gamma = compute_Gamma(T, omega_hat_k)
        P_pred = Phi_k @ P[-1] @ Phi_k.T + Gamma @ Cov_u @ Gamma.T

        return X_hat_pred, P_pred
    def correction(z):
        X_hat_pred, P_pred = prediction()
        x_hat_pred, y_hat_pred, omega_hat_pred = X_hat_pred[0][0], X_hat_pred[1][0], X_hat_pred[2][0]
        # Kalman gain
        H_k = compute_H_k(x_hat_pred, y_hat_pred)
        S_k = H_k @ P_pred @ H_k.T + W_k
        K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)
        
        # Update
        X_hat_k1 = X_hat_pred + K_k @ (z - H_k @ X_hat_pred)
        P_k1 = (np.eye(3) - K_k @ H_k) @ P_pred

        X_hat.append(X_hat_k1)
        P.append(P_k1)

    correction(z)

z_meas = np.loadtxt('./Kalmannonlin.txt')
for z in z_meas:
    step(z)


data = np.array(X_hat)[1:].reshape(-1, 3)
covariances = np.array(P).reshape(-1, 3, 3) 
vrai = np.loadtxt('./etatvrai.txt')

def plot_error():
    X = data[:, 0]
    X_vrai = vrai[0]
    Y = data[:, 1]
    Y_vrai = vrai[1]
    omega = data[:, 2]
    omega_vrai = vrai[2]

    _X_hat = np.array([X, Y, omega])
    _X = np.array([X_vrai, Y_vrai, omega_vrai])
    error = np.linalg.norm(_X - _X_hat, axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(error)
    ax.set_title("Evolution of the Quadratic Estimation Error")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quadratic Estimation Error")
    ax.grid(alpha=0.3)
    plt.savefig('error.png')

def plot_estimates():
    X = data[:, 0]
    X_vrai = vrai[0]
    Y = data[:, 1]
    Y_vrai = vrai[1]
    omega = data[:, 2]
    omega_vrai = vrai[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    axs[0].plot(X, label='Estimated')
    axs[0].plot(X_vrai, label='True Value')
    axs[0].set_ylabel('X')
    axs[0].set_title('X over Measures')

    axs[1].plot(Y, label='Estimated')
    axs[1].plot(Y_vrai, label='True Value')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Y over Measures')

    axs[2].plot(omega, label='Estimated')
    axs[2].plot(omega_vrai, label='True Value')
    axs[2].set_ylabel('Omega')
    axs[2].set_title('Omega over Measures')
    axs[2].set_xlabel('Measure')

    for i in range(3):
        axs[i].legend()

def plot_trace():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    trace = []
    for i in range(301):
        trace.append(np.trace(covariances[i]))
    ax.plot(trace)
    ax.set_title("Evolution of the trace of the covariance matrix")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Trace of Covariance Matrix")
    ax.grid(alpha=0.3)
    plt.savefig('trace.png')

# plot_estimates()
plt.tight_layout()
plot_trace()
plot_error()
# plt.show()
