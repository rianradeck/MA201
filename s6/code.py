import numpy as np

# Paramètres donnés
T = 0.01
sigma_v = 0.1
sigma_omega = 0.01
sigma_w = 0.2
V = 3.0  # m/s
omega = 2 * np.pi / 3  # rad/s
x_b, y_b = 2, 5  # Coordonnées du point fixe
x0 = np.array([[0, 0, 0]]).T  # État initial estimé
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

# Filtre de Kalman étendu
def extended_kalman_filter(z_meas, x0, P0):
    # Initialisation
    x_hat = x0
    P_k = P0
    estimates = []
    covariances = []

    for z_k in z_meas:
        # Prédiction de l'état
        phi_hat = x_hat[2][0]
        # x_hat_pred = compute_Phi_k(V, T, phi_hat) @ x_hat
        x_hat_pred = np.array([
            [x_hat[0][0] + V * T * np.cos(phi_hat)],
            [x_hat[1][0] + V * T * np.sin(phi_hat)],
            [x_hat[2][0] + T * omega]
        ])
        # x_hat_pred => x_hat_k+1_sachant_k (3, 1)


        # Prédiction de la covariance (3, 3)
        Phi_k = compute_Phi_k(V, T, phi_hat)
        Gamma = compute_Gamma(T, phi_hat)
        P_k_pred = Phi_k @ P_k @ Phi_k.T + Gamma @ Cov_u @ Gamma.T
        
        # Calcul de H_k (1, 3)
        H_k = compute_H_k(x_hat_pred[0][0], x_hat_pred[1][0])
        
        # Gain de Kalman (3, 1)
        S_k = (H_k @ P_k_pred @ H_k.T) + W_k
        K_k = P_k_pred @ H_k.T @ np.linalg.inv(S_k)
        
        # Mise à jour de l'estimation de l'état
        x_hat = x_hat_pred + K_k @ (z_k - H_k @ x_hat_pred)
        
        # Mise à jour de la covariance
        P_k = (np.eye(3) - K_k @ H_k) @ P_k_pred
        
        # Sauvegarde de l'estimation
        estimates.append(x_hat)
        covariances.append(P_k)
    
    return np.array(estimates), np.array(covariances)

z_meas = np.loadtxt('./Kalmannonlin.txt')
for z in z_meas:
    step(z)

import matplotlib.pyplot as plt

data = np.array(X_hat).reshape(-1, 3)
covariances = np.array(P).reshape(-1, 3, 3) 
vrai = np.loadtxt('./etatvrai.txt')

def plot_estimates():
    X = data[:, 0]
    X_vrai = vrai[0]
    Y = data[:, 1]
    Y_vrai = vrai[1]
    omega = data[:, 2]
    omega_vrai = vrai[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    axs[0].plot(X, label='Estimé')
    axs[0].plot(X_vrai, label='Vrai')
    axs[0].set_ylabel('X')
    axs[0].set_title('X over Measures')

    axs[1].plot(Y, label='Estimé')
    axs[1].plot(Y_vrai, label='Vrai')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Y over Measures')

    axs[2].plot(omega, label='Estimé')
    axs[2].plot(omega_vrai, label='Vrai')
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
    ax.set_title("Trace by iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Trace")
    ax.grid(alpha=0.3)

plot_trace()
plt.tight_layout()
plt.show()
