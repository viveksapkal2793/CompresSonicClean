# Compressive Sensing and Denoising of Audio Signals

A comprehensive implementation of compressive sensing techniques for audio signal recovery, with enhanced post-processing denoising.

## Project Overview

This project demonstrates how compressed sensing (CS) can be used to efficiently compress, store, and recover audio signals using significantly fewer samples than traditionally required by the Nyquist-Shannon sampling theorem. The implementation includes:

- Compression of audio signals to approximately 40% of their original size
- Recovery using L1-minimization (sparsity-promoting optimization)  
- Advanced frame-based denoising for enhanced audio quality
- Visualization tools to compare original and recovered signals

## Theory

Compressed sensing relies on two key principles:
1. **Sparsity**: Audio signals have sparse representations in certain domains (e.g., DCT, Fourier)
2. **Incoherence**: Random measurement matrices provide efficient signal acquisition

Using these principles, we can:
- Represent a signal with far fewer measurements than traditional methods require
- Recover the original signal using optimization techniques that promote sparsity

## Implementation Features

### Compression
- Random measurement matrices (Bernoulli or Gaussian)
- Adjustable compression ratio (default: 40%)
- Serialized storage of compressed measurements

### Recovery
- DCT-domain sparsity representation
- L1-minimization via Lasso regression
- Quality assessment using PSNR metrics

### Denoising
- Frame-based processing with overlapping windows
- Window functions (Hamming, Blackman) for artifact reduction
- Primal-dual gradient descent optimization
- Soft thresholding for enhanced sparsity

## Usage

### Audio Compression and Recovery

```bash
python cs_audio.py
```
This will:
1. Process all audio files in the input directory
2. Save compressed measurements to compressed
3. Save recovered audio to recovered
4. Generate analysis plots for each file

### Audio Denoising

```bash
python denoising.py
```
This will:
1. Process all recovered audio files in the recovered directory
2. Apply advanced denoising algorithms
3. Save enhanced audio to denoised
4. Generate comparison plots

### Audio Playback

```bash
python listen.py <filename.wav>
```
Plays back any processed audio file. Running without arguments will list available files.

## File Structure

- cs_audio.py: Main implementation of audio compression and recovery
- denoising.py: Advanced denoising algorithms for enhanced quality
- compressed_sensing.py: Basic example using synthetic sine waves
- listen.py: Utility for audio playback and validation
- requirements.txt: Project dependencies

## Results

The implementation demonstrates:
- Successful compression of audio to 40% of original size
- Recovery with minimal perceptual quality loss
- Further quality enhancement through specialized denoising
- Visual analysis of time and frequency domains

## Installation

1. Clone the repository:
```bash
git clone https://github.com/viveksapkal2793/CompresSonicClean.git
cd CompresSonicClean
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create an input directory and add audio files (WAV, MP3, OGG, FLAC, M4A):
```bash
mkdir input
# Copy your audio files to the input directory
```

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- librosa (for audio processing)
- soundfile (for audio I/O)
- sounddevice (for audio playback)
