import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso

USE_GAUSSIAN = False  # Change to False to use Bernoulli measurement matrix

def plot_fft(signal, fs, title, ax, show_40hz_marker=True):
    N = len(signal)
    signal_fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Plot only positive frequencies
    ax.plot(freqs[:N//2], np.abs(signal_fft)[:N//2])
    
    # Set x-axis to logarithmic scale
    ax.set_xscale('log')
    
    # Set x-axis ticks at multiples of 10 for logarithmic scale
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1000'])
    
    if show_40hz_marker:
        # Add a vertical line at 40Hz
        ax.axvline(x=40, color='r', linestyle='--', label='40 Hz')
        ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    
    ax.grid(True, which="both", ls="--")

# Step 1: Generate the original signal (40Hz sine wave)
fs = 1000  # Original sampling frequency (1000Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
f = 40  # Frequency of the sine wave (40Hz)
x = np.sin(2 * np.pi * f * t)

# Step 2: Adjust sampling rate below Nyquist
fs_low = 50  # Reduced sampling frequency (adjust this value as needed)
n_samples = int(fs_low)  # Number of samples
sample_indices = np.linspace(0, len(x) - 1, n_samples, dtype=int)
x_sampled = x[sample_indices]
t_sampled = t[sample_indices]

# Step 3: Compressed sensing
if USE_GAUSSIAN:
    # Create a random Gaussian measurement matrix
    np.random.seed(0)
    A = np.random.randn(n_samples, len(x))
else:
    # Create a random Bernoulli measurement matrix
    np.random.seed(0)
    A = np.random.choice([1, -1], size=(n_samples, len(x)))

# Perform measurement
y = A @ x

# Step 4: Signal recovery using L1 minimization (Lasso)
# DCT basis
D = dct(np.eye(len(x)), norm='ortho')

# Solve Lasso: min ||y - A @ D @ alpha||_2 + lambda * ||alpha||_1
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(A @ D, y)
x_recovered = idct(lasso.coef_, norm='ortho')

# Step 5: Plot results
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot Original Signal
axes[0, 0].plot(t, x, label='Original Signal (40Hz sine wave)')
axes[0, 0].legend()
axes[0, 0].set_title('Original Signal')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')

# Plot FFT of Original Signal
plot_fft(x, fs, 'FFT of Original Signal', axes[0, 1])

# Plot Sampled Signal
axes[1, 0].stem(t_sampled, x_sampled, 'r', markerfmt='ro', basefmt=" ", label='Sampled Signal (fs = {}Hz)'.format(fs_low))
axes[1, 0].legend()
axes[1, 0].set_title('Sampled Signal')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Amplitude')

# Plot FFT of Sampled Signal (no 40 Hz marker)
plot_fft(x_sampled, fs_low, 'FFT of Sampled Signal', axes[1, 1], show_40hz_marker=False)

# Plot Recovered Signal
axes[2, 0].plot(t, x_recovered, 'g', label='Recovered Signal (Compressed Sensing)')
axes[2, 0].legend()
axes[2, 0].set_title('Recovered Signal')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Amplitude')

# Plot FFT of Recovered Signal
plot_fft(x_recovered, fs, 'FFT of Recovered Signal', axes[2, 1])

# Adjust layout and show plot
plt.tight_layout()
plt.show()
