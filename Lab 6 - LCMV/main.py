import numpy as np
import matplotlib.pyplot as plt

d = 0.5 # half wavelength spacing
Nr = 8 # number of elements in the array

theta_jammer = np.deg2rad(-20) # direction of arrival (feel free to change this, it's arbitrary)
sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the jamming signal
t = np.arange(N)/sample_rate # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_jammer))
s = s.reshape(-1,1) # make s a column vector (8x1)
tx = tx.reshape(1,-1) # make tx a row vector (1x10000)

# Simulate the received signal X through a matrix multiply
X = s @ tx 
print(X.shape) # 8x10000.  X is now going to be a 2D array, 1D is time and 1D is the spatial dimension
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # X and n are both 8x10000

# Let's point at the SOI at 15 deg, and another potential SOI that we didn't actually simulate at 60 deg
soi1_theta = 15 / 180 * np.pi # convert to radians
soi2_theta = 60 / 180 * np.pi

# LCMV weights
R_inv = np.linalg.pinv(np.cov(X)) # 8x8
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
C = np.concatenate((s1, s2), axis=1) # 8x2
f = np.ones(2).reshape(-1,1) # 2x1

# LCMV equation
#    8x8   8x2                    2x8        8x8   8x2  2x1
w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # output is 8x1

w = w.squeeze() # remove the extra dimension (3x1 to 3)

# Visualize the beam pattern (don't worry as much about this part)
N_fft = 1024
w = np.conj(w) # or else our answer will be negative/inverted
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot([theta_jammer, theta_jammer], [-30, 5], 'r--')
ax.plot([soi1_theta, soi1_theta], [-30, 5], 'g--')
ax.plot([soi2_theta, soi2_theta], [-30, 5], 'g--')
ax.set_theta_zero_location('N') # type: ignore # make 0 degrees point up
ax.set_theta_direction(-1) # type: ignore # increase clockwise
ax.set_thetamin(-90) # type: ignore # only show top half
ax.set_thetamax(90) # type: ignore
ax.set_ylim((-30, 5)) # because there's no noise, only go down 30 dB
plt.show()
