import numpy as np
import pandas as pd
from utils import THYMIO_WIDTH_MM

L = THYMIO_WIDTH_MM

# ======================================================
# COMPUTE R  (ROBOT IMMOBILE → BRUIT CAMÉRA PUR)
# ======================================================
def compute_R(csv_path):

    df = pd.read_csv(csv_path)
    df = df.dropna()   # remove rows without ArUco

    xs = df["x_cam"].astype(float).to_numpy()
    ys = df["y_cam"].astype(float).to_numpy()
    thetas = df["theta_cam"].astype(float).to_numpy()

    # Angle wrapping
    thetas = (thetas + np.pi) % (2*np.pi) - np.pi

    var_x = np.var(xs, ddof=1)
    var_y = np.var(ys, ddof=1)

    # circular variance for theta
    mean_theta = np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas)))
    diff_theta = (thetas - mean_theta + np.pi) % (2*np.pi) - np.pi
    var_theta = np.var(diff_theta, ddof=1)

    R = np.diag([var_x, var_y, var_theta])
    return R


# ======================================================
# COMPUTE Q (ROBOT EN MOUVEMENT → ERREUR DU MODELE)
# ======================================================
def g_motion(x, control, dt):
    vL, vR = control
    v = (vR + vL)/2
    w = (vR - vL)/L

    xp = x.copy()
    xp[0] += v * dt * np.cos(x[2])
    xp[1] += v * dt * np.sin(x[2])
    xp[2] += w * dt
    xp[2] = (xp[2] + np.pi) % (2*np.pi) - np.pi
    return xp

def compute_Q(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()    # keep only rows with camera data
    df = df.sort_values("timestamp").reset_index(drop=True)

    residuals = []

    for k in range(len(df)-1):
        xk = np.array([df.loc[k,'x_cam'], df.loc[k,'y_cam'], df.loc[k,'theta_cam']], float)
        uk = np.array([df.loc[k,'vL'], df.loc[k,'vR']], float)

        t0 = df.loc[k,'timestamp']
        t1 = df.loc[k+1,'timestamp']
        dt = t1 - t0
        if dt <= 0 or dt > 0.2:
            continue

        xkp1_true = np.array([df.loc[k+1,'x_cam'], df.loc[k+1,'y_cam'], df.loc[k+1,'theta_cam']], float)

        x_pred = g_motion(xk, uk, dt)
        diff = xkp1_true - x_pred
        diff[2] = (diff[2] + np.pi) % (2*np.pi) - np.pi

        residuals.append(diff)

    residuals = np.vstack(residuals)
    Q = np.cov(residuals.T, ddof=1)
    return Q


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    path = input("path : log_data.csv")

    print("\n Computing R (camera noise)…")
    R = compute_R(path)
    print("\nR =\n", R)
    np.save("R.npy", R)

    #print("\n Computing Q (process noise)…")
    #Q = compute_Q(path)
    #print("\nQ =\n", Q)
    #np.save("Q.npy", Q)

    print("\n Saved R.npy and Q.npy")
