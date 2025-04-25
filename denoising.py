import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io.wavfile import write
import warnings

# Only suppress specific warnings that don't affect algorithm performance
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val < 1e-10:  # Avoid division by near-zero
        return np.zeros_like(data)
    return data/max_val

def blackman(x):
    # Blackman window
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    N = x.shape[-1]
    n = np.arange(N)
    w = a0 - a1*np.cos(2*np.pi*n/N) - a2*np.cos(4*np.pi*n/N)
    return w*x

def hamming(x):
    # Hamming window
    a0 = 0.53836
    a1 = 0.46164
    N = x.shape[-1]
    n = np.arange(N)
    w = a0 - a1*np.cos(2*np.pi*n/N)
    return w*x

def denoise_audio(filename, output_dir, plot_results=False):
    print(f"Processing: {os.path.basename(filename)}")
    
    # Load audio file
    data, fs = sf.read(filename)
    
    # Handle mono/stereo format
    if len(data.shape) > 1 and data.shape[1] > 1:
        print("Converting stereo to mono")
        data = data.mean(axis=1)  # Convert to mono
    
    # Keep original parameters - critical for good results
    num_samples = data.shape[-1]
    n = 1024
    m = 256
    s = 32
    zero_thresh = 0.1
    min_sparse = 10
    snr_db = -100
    eps = 1e-8
    alpha = 0.1
    beta = 0.5
    inc = n//2
    snr = 10**(snr_db/10)

    # Making length of data a multiple of n
    rem = len(data) % n
    if rem > 0:
        data = data[0:len(data)-rem]
    data_clean = np.zeros(len(data))

    # F is full Fourier matrix
    # M is a random sampling matrix
    # A is random sensing matrix
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    ii = sorted(rng.choice(n, size=m, replace=False))
    M = np.zeros((m, n))
    j = 0
    for i in ii:
        M[j, i] = 1
        j += 1

    F = np.array([[np.exp(2j*np.pi*i*j/n)/n for i in range(n)] for j in range(n)])
    A = M.dot(F)

    def get_sparse(x):
        # Create a copy to avoid modifying the original
        x_copy = x.copy()
        x_copy[np.where(np.abs(x_copy) <= zero_thresh)] = 0
        if len(np.where(np.abs(x_copy) > zero_thresh)[0]) < min_sparse:
            return 0
        return np.fft.ifft(x_copy).real

    # Keep the original functions but add minimal safety
    def f(x, y, ii, lm):
        ''' Objective function '''
        xt = np.fft.ifft(x)
        vec = xt[ii]
        T1 = np.sum((np.abs(vec - y))**2)
        T2 = np.sum(np.abs(x)**2)/n/snr
        T3 = np.sum(np.abs(x))
        return lm*(T1 - T2) + T3

    def shrink(x):
        val = np.zeros(x.shape, dtype=np.complex128)
        for i in range(x.shape[-1]):
            if np.abs(x[i]) > 1e-9:
                val[i] = x[i]/abs(x[i])
        return val

    def grad_f_x(x, y, ii, lm):
        ''' Gradient of objective, lm ==> lagrange multiplier '''
        xt = np.fft.ifft(x)
        vec = xt[ii]
        res = vec - y
        vec = np.zeros(x.shape, dtype=np.complex128)
        vec[ii] = res
        vec = np.fft.fft(vec)/n
        return lm*(2*vec - (2/n/snr)*x) + shrink(x)

    def grad_f_lm(x, y, ii):
        ''' Gradient of objective, lm ==> lagrange multiplier '''
        xt = np.fft.ifft(x)
        vec = xt[ii]
        T1 = np.sum((np.abs(vec- y))**2)
        T2 = np.sum(np.abs(x)**2)/n/snr
        return (T1 - T2)

    def bls_x(x, y, ii, lm, del_x, alpha, beta):
        ''' Backtracking line search '''
        t = 1
        ff = f(x, y, ii, lm)
        xx = np.sum(np.abs(del_x)**2)
        
        # Add a maximum iteration count to avoid infinite loops
        max_iters = 50
        iters = 0
        
        while iters < max_iters:
            try:
                if f(x + t*del_x, y, ii, lm) <= ff-alpha*t*xx:
                    break
            except:
                pass  # Skip errors but continue loop
            
            t = beta * t
            iters += 1
            
        return t

    def bls_lm(x, y, ii, lm, del_lm, alpha, beta):
        ''' Backtracking line search '''
        t = 1
        ff = f(x, y, ii, lm)
        ll = del_lm**2
        
        # Add a maximum iteration count to avoid infinite loops
        max_iters = 50
        iters = 0
        
        while iters < max_iters:
            try:
                if f(x, y, ii, lm + t*del_lm) >= ff + alpha*t*ll:
                    break
            except:
                pass  # Skip errors but continue loop
                
            t = beta * t
            iters += 1
            
        return t

    # Process each frame
    i = 0
    total_frames = (len(data)-n) // inc
    while i < len(data)-n:
        try:
            y = M.dot(hamming(data[int(i):int(i+n)]))
            x = np.fft.fft(hamming(data[int(i):int(i+n)]))
            lm = 0  # Start with lambda = 0 as in original code
            
            # Primal-Dual Gradient Descent with Line Search for each frame
            step = 0
            
            current_frame = int(i*(n/inc)/n) + 1  # +1 for 1-based counting
            print(f"Processing frame {current_frame}/{total_frames}", end='\r')

            while step < 1000:  # Keep original iteration count
                step += 1
                
                # Compute direction
                del_x = -grad_f_x(x, y, ii, lm)
                
                # Line search
                t1 = bls_x(x, y, ii, lm, del_x, alpha, beta)

                # Check convergence
                try:
                    if np.max(np.abs((t1*del_x))) < eps:
                        break
                except:
                    break
                
                # Update primal variable
                x = x + t1*del_x
                
                # Compute dual direction
                del_lm = grad_f_lm(x, y, ii)
                
                # Line search
                t2 = bls_lm(x, y, ii, lm, del_lm, alpha, beta)

                # Update dual variable
                if lm + t2*del_lm >= 0:
                    lm = lm + t2*del_lm

            # Extract sparse representation
            frame_clean = get_sparse(x)
            
            # Only add valid values to output
            if isinstance(frame_clean, np.ndarray):
                data_clean[int(i):int(i+n)] += frame_clean
                
        except Exception as e:
            # Skip frame on error
            print(f"\nSkipping frame {current_frame} due to error: {str(e)}")
        
        # Move to next frame
        i += inc
    
    # Replace any NaN/Inf with zeros in output
    data_clean = np.nan_to_num(data_clean)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(output_dir, f"denoised_{base_name}.wav")
    
    # Save denoised audio - keep original scale factor
    write(output_filename, fs, 0.7*normalize(data_clean))
    print(f"\nSaved: {output_filename}")
    
    # Plot if requested
    if plot_results:
        plt.figure(figsize=(12, 6))
        tt = np.arange(data_clean.size)/fs
        plt.title(f'Original vs Denoised: {base_name}')
        plt.plot(tt, normalize(data), label='Original')
        plt.plot(tt, normalize(data_clean), alpha=0.7, label='Denoised')
        plt.ylabel('Normalized Signal Value')
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        # Save plot
        plt_path = os.path.join(output_dir, f"denoised_{base_name}_plot.png")
        plt.savefig(plt_path)
        plt.close()
        print(f"Saved plot: {plt_path}")
    
    return True

def process_directory():
    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    recovered_dir = os.path.join(current_dir, 'recovered')
    denoised_dir = os.path.join(current_dir, 'denoised')
    
    # Ensure directories exist
    ensure_directory_exists(denoised_dir)
    
    if not os.path.exists(recovered_dir):
        print(f"Recovered directory not found: {recovered_dir}")
        return
    
    # Get all WAV files
    wav_files = []
    for file in os.listdir(recovered_dir):
        if file.lower().endswith('.wav'):
            wav_files.append(os.path.join(recovered_dir, file))
    
    if not wav_files:
        print(f"No WAV files found in {recovered_dir}")
        return
    
    # Process each file
    print(f"Found {len(wav_files)} files to process")
    success_count = 0
    
    for i, file_path in enumerate(wav_files):
        print(f"\n[{i+1}/{len(wav_files)}] Processing {os.path.basename(file_path)}")
        if denoise_audio(file_path, denoised_dir, plot_results=True):
            success_count += 1
    
    print(f"\nDenoising complete: {success_count}/{len(wav_files)} files processed successfully")
    print(f"Denoised files saved to: {denoised_dir}")

if __name__ == "__main__":
    print("Audio Denoising")
    print("==============")
    process_directory()