WHEEL_RADIUS_MM = 20
THYMIO_WIDTH_MM = 120
#THYMIO2MMS = 0.4347826
THYMIO2MMS = 0.351

import numpy as np
try:
    Q = np.load("Q.npy")
    R = np.load("R.npy")
    print("Q:", Q)
    print("R:", R)
except FileNotFoundError as e:
    print(f"ERROR: {e}")  # <-- This is likely your problem!