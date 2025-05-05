import numpy as np
import matplotlib.pyplot as plt

# Array params
center_freq = 3.3e9
sample_rate = 30e6
d = 0.045 * center_freq / 3e8
print("d:", d)

# Load "training data" which is just A and B, then calc Rinv
filename = '3p3G_A_B.npy'
X_A_B = np.load(filename)
R_training = X_A_B @ X_A_B.conj().T # Calc covariance matrix
Rinv_training = np.linalg.pinv(R_training)

# Includes all three signals, we'll call C our SOI
filename = '3p3G_A_B_C.npy'
X = np.load(filename)
Nr = X.shape[0]

max_angle = -0.34073979726153913 # found in step 1, this is the DOA of C in radians

# Calc MVDR weights towards C using training Rinv
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # steering vector in the desired direction theta
s = s.reshape(-1,1) # make into a column vector (size 3x1)
w = (Rinv_training @ s)/(s.conj().T @ Rinv_training @ s) # MVDR/Capon equation

# Visualize the beam pattern
N_fft = 1024
w = w.squeeze() # remove the extra dimension (size 3x1 -> size 3)
w = np.conj(w) # or else our answer will be negative/inverted
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians)
plt.plot(np.rad2deg(theta_bins), w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
plt.axvline(x=np.rad2deg(max_angle), color='green', linestyle='--', label='max angle')
plt.grid()
plt.xlabel('Angle (degrees)')
plt.ylabel('Power (dB)')
plt.show()

