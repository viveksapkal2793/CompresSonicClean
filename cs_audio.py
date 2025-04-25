import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso
import librosa
import soundfile as sf
import os
import pickle

USE_GAUSSIAN = False
COMPRESSION_RATIO = 0.4  # Percentage of samples to use (20%)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def plot_fft(signal, fs, title, ax, show_marker=False):
    N = len(signal)
    signal_fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Plot only positive frequencies
    ax.plot(freqs[:N//2], np.abs(signal_fft)[:N//2])
    
    # Set x-axis to logarithmic scale
    ax.set_xscale('log')
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(['10', '100', '1k', '10k'])
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, which="both", ls="--")

def process_audio(file_path, compressed_dir, recovered_dir, segment_duration=2.0):
    """
    Process audio using compressed sensing and save results to appropriate directories.
    
    Args:
        file_path: Path to input audio file
        compressed_dir: Directory to save compressed measurements
        recovered_dir: Directory to save recovered audio
        segment_duration: Duration in seconds to process
    """
    try:
        # Load audio file (mono conversion if stereo)
        x_full, fs = librosa.load(file_path, sr=None, mono=True)
        
        # Get filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Processing: {base_name}")
        print(f"Sample rate: {fs} Hz, Duration: {len(x_full)/fs:.2f}s")
        
        # Take a segment for demonstration
        segment_samples = int(segment_duration * fs)
        if len(x_full) > segment_samples:
            x = x_full[:segment_samples]
            print(f"Using first {segment_duration}s segment")
        else:
            x = x_full
            print(f"Using entire file ({len(x_full)/fs:.2f}s)")
        
        t = np.linspace(0, len(x)/fs, len(x), endpoint=False)
        
        # Compressed sensing
        n_samples = int(COMPRESSION_RATIO * len(x))
        
        if USE_GAUSSIAN:
            np.random.seed(0)
            A = np.random.randn(n_samples, len(x))
        else:
            np.random.seed(0)
            A = np.random.choice([1, -1], size=(n_samples, len(x)))
        
        # Perform measurement
        y = A @ x
        
        # Save compressed measurements
        compressed_data = {
            'measurements': y,
            'matrix_type': 'gaussian' if USE_GAUSSIAN else 'bernoulli',
            'matrix_seed': 0,
            'compression_ratio': COMPRESSION_RATIO,
            'fs': fs,
            'original_length': len(x)
        }
        
        compressed_path = os.path.join(compressed_dir, f"{base_name}_compressed.pkl")
        with open(compressed_path, 'wb') as f:
            pickle.dump(compressed_data, f)
        
        print(f"Saved compressed data: {compressed_path} ({len(y)} measurements)")
        
        # Signal recovery using L1 minimization
        D = dct(np.eye(len(x)), norm='ortho')
        
        print("Recovering signal...")
        lasso = Lasso(alpha=0.001, max_iter=10000)
        lasso.fit(A @ D, y)
        x_recovered = idct(lasso.coef_, norm='ortho')
        
        # Save recovered audio
        recovered_path = os.path.join(recovered_dir, f"{base_name}_recovered.wav")
        sf.write(recovered_path, x_recovered, fs)
        print(f"Saved recovered audio: {recovered_path}")
        
        # Plotting
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Original audio
        axes[0, 0].plot(t, x)
        axes[0, 0].set_title('Original Audio')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        plot_fft(x, fs, 'FFT of Original Audio', axes[0, 1])
        
        # Compressed measurements
        axes[1, 0].stem(range(n_samples), y, 'r', markerfmt='ro', basefmt=" ")
        axes[1, 0].set_title(f'Compressed Measurements ({n_samples} samples, {COMPRESSION_RATIO*100:.1f}%)')
        axes[1, 0].set_xlabel('Measurement Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 1].text(0.5, 0.5, "Measurements not in time domain", ha='center', va='center')
        axes[1, 1].set_title('Compressed Domain')
        
        # Recovered audio
        axes[2, 0].plot(t, x_recovered, 'g')
        axes[2, 0].set_title('Recovered Audio')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        plot_fft(x_recovered, fs, 'FFT of Recovered Audio', axes[2, 1])
        
        # Add reconstruction quality metrics
        mse = np.mean((x - x_recovered)**2)
        psnr = 10 * np.log10(np.max(x)**2 / mse) if mse > 0 else float('inf')
        fig.suptitle(f'Audio Compressed Sensing - {base_name} (PSNR: {psnr:.2f} dB)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plot
        plt_path = os.path.join(recovered_dir, f"{base_name}_plot.png")
        plt.savefig(plt_path)
        print(f"Saved plot: {plt_path}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(input_dir):
    """Process all audio files in the input directory"""
    
    # Create output directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    compressed_dir = os.path.join(current_dir, 'compressed')
    recovered_dir = os.path.join(current_dir, 'recovered')
    
    ensure_directory_exists(compressed_dir)
    ensure_directory_exists(recovered_dir)
    
    # Make sure input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Get all supported audio files
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(input_dir, file))
    
    if not audio_files:
        print(f"No supported audio files found in {input_dir}")
        return
    
    # Process each file
    print(f"Found {len(audio_files)} audio files")
    success_count = 0
    
    for i, file_path in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing {os.path.basename(file_path)}")
        if process_audio(file_path, compressed_dir, recovered_dir):
            success_count += 1
    
    print(f"\nProcessing complete: {success_count}/{len(audio_files)} files processed successfully")
    print(f"Compressed files saved to: {compressed_dir}")
    print(f"Recovered files saved to: {recovered_dir}")

if __name__ == "__main__":
    print("Audio Compressed Sensing")
    print("========================")
    
    # input_dir = input("Enter the path to the input audio directory: ").strip()
    input_dir = "./input" 

    # Use default if empty
    if not input_dir:
        input_dir = "./input"
        print(f"Using default input directory: {input_dir}")
    
    process_directory(input_dir)