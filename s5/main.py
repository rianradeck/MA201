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

def plot(data, pred = None, cov = None, remove_k_first=0, remove_k_last=0):
    N = data.shape[1]

    X = data[0, :]
    Y = data[1, :]
    Z = data[2, :]
    Vx = data[3, :]
    Vy = data[4, :]
    Vz = data[5, :]
    if pred is not None:
        pred = np.array(pred)
        X_pred = pred[0, remove_k_first:-remove_k_last]
        Y_pred = pred[1, remove_k_first:-remove_k_last]
        Z_pred = pred[2, remove_k_first:-remove_k_last]
        Vx_pred = pred[3, remove_k_first:-remove_k_last]
        Vy_pred = pred[4, remove_k_first:-remove_k_last]
        Vz_pred = pred[5, remove_k_first:-remove_k_last]
    if cov is not None:
        cov = np.array(cov)
        trace = np.trace(cov, axis1=1, axis2=2)
        sigma_x = np.sqrt(cov[remove_k_first:-remove_k_last, 0, 0])
        sigma_y = np.sqrt(cov[remove_k_first:-remove_k_last, 1, 1])
        sigma_z = np.sqrt(cov[remove_k_first:-remove_k_last, 2, 2])
        sigma_vx = np.sqrt(cov[remove_k_first:-remove_k_last, 3, 3])
        sigma_vy = np.sqrt(cov[remove_k_first:-remove_k_last, 4, 4])
        sigma_vz = np.sqrt(cov[remove_k_first:-remove_k_last, 5, 5])

    def plot_in_time(axis, data, T=T, label = "", title="", ylabel="", xlabel="", offset=0):
        axis.plot(T * np.arange(offset, offset + len(data)), data, label=label)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel)
    
    def fill_between_in_time(axis, data, sigma, T=T, label="", offset=0):
        axis.fill_between(T * np.arange(offset, offset + len(data)), data - 2*sigma, data + 2*sigma, alpha=0.3, label=label)

    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    zmX = axes[0, 0].inset_axes([0.65, 0.10, 0.3, 0.35]) # x0, y0, width, height
    zoom_limits = (40, 50)
    plot_in_time(zmX, X[zoom_limits[0]:zoom_limits[1]+1], label='Real', title='Zoomed X Position by time', ylabel='X Position', xlabel='Time', offset=zoom_limits[0])

    plot_in_time(axes[0, 0], X, label="Real", title='True X Position by time', ylabel='X Position', xlabel='Time')
    plot_in_time(axes[0, 1], Vx, label="Real", title='True X Velocity by time', ylabel='X Velocity', xlabel='Time')
    plot_in_time(axes[1, 0], Y, label="Real", title='True Y Position by time', ylabel='Y Position', xlabel='Time')    
    plot_in_time(axes[1, 1], Vy, label="Real", title='True Y Velocity by time', ylabel='Y Velocity', xlabel='Time')
    plot_in_time(axes[2, 0], Z, label="Real", title='True Z Position by time', ylabel='Z Position', xlabel='Time')
    plot_in_time(axes[2, 1], Vz, label="Real", title='True Z Velocity by time', ylabel='Z Velocity', xlabel='Time')

    if pred is not None:
        X_vals = list(range(remove_k_first, N+1))
        plot_in_time(axes[0, 0], X_pred, label='Prediction', title='Predicted X Position by time', ylabel='X Position', xlabel='Time', offset=remove_k_first)
        plot_in_time(zmX, X_pred[zoom_limits[0]-1: zoom_limits[1]], label='Prediction', offset=zoom_limits[0])
        plot_in_time(axes[0, 1], Vx_pred, label='Prediction', title='Predicted X Velocity by time', ylabel='X Velocity', xlabel='Time', offset=remove_k_first)
        plot_in_time(axes[1, 0], Y_pred, label='Prediction', title='Predicted Y Position by time', ylabel='Y Position', xlabel='Time', offset=remove_k_first)
        plot_in_time(axes[1, 1], Vy_pred, label='Prediction', title='Predicted Y Velocity by time', ylabel='Y Velocity', xlabel='Time', offset=remove_k_first)
        plot_in_time(axes[2, 0], Z_pred, label='Prediction', title='Predicted Z Position by time', ylabel='Z Position', xlabel='Time', offset=remove_k_first)
        plot_in_time(axes[2, 1], Vz_pred, label='Prediction', title='Predicted Z Velocity by time', ylabel='Z Velocity', xlabel='Time', offset=remove_k_first)

    if cov is not None:
        fill_between_in_time(axes[0, 0], X_pred, sigma_x, label='Confidence', offset=remove_k_first)
        fill_between_in_time(zmX, X_pred[zoom_limits[0]-1: zoom_limits[1]], sigma_x[zoom_limits[0]-1: zoom_limits[1]], label='Confidence', offset=zoom_limits[0])
        fill_between_in_time(axes[0, 1], Vx_pred, sigma_vx, label='Confidence', offset=remove_k_first)
        fill_between_in_time(axes[1, 0], Y_pred, sigma_y, label='Confidence', offset=remove_k_first)
        fill_between_in_time(axes[1, 1], Vy_pred, sigma_vy, label='Confidence', offset=remove_k_first)
        fill_between_in_time(axes[2, 0], Z_pred, sigma_z, label='Confidence', offset=remove_k_first)
        fill_between_in_time(axes[2, 1], Vz_pred, sigma_vz, label='Confidence', offset=remove_k_first)
        
    def plot_trace():
        fig_, axes_ = plt.subplots(1, 3, figsize=(18, 6))
        if cov is not None:
            # axes_.set_title('Trace of Cov Matrix')
            cut = 25
            axes_[0].semilogy(trace, label='Trace of P_k')
            axes_[1].semilogy(range(cut), trace[:cut], label=f'Trace < {cut}')
            axes_[2].semilogy(range(cut, len(trace)), trace[cut:], label=f'Trace >= {cut}')

            for i in range(3):
                axes_[i].set_xlabel('k')
                axes_[i].set_ylabel('Trace')
                axes_[i].grid(alpha=0.3)
                axes_[i].legend()

    for i in range(3):
        for j in range(2):
            axes[i, j].legend()
            axes[i, j].grid(alpha=0.3)
    zmX.grid(alpha=0.3)
    axes[0, 0].indicate_inset_zoom(zmX, edgecolor='black')

    # plot_trace()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig('plot.png')

for measure in measures:
    step(measure)
x_hat = np.array(x_hat).T.reshape(6, -1)
plot(real_data, x_hat, P, 1, 1)