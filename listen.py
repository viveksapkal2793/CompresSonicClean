import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

def validate_audio(data):
    """Validate and fix audio data"""
    if np.isnan(data).any() or np.isinf(data).any():
        print("Warning: Audio file contains invalid values. Fixing...")
        return np.nan_to_num(data)  # Replace NaN with 0 and Inf with finite values
    
    # Check if completely silent/corrupted
    if np.max(np.abs(data)) < 1e-6:
        print("Warning: Audio appears to be completely silent or corrupted.")
    
    return data

def play_audio(filename):
    """Play audio file with validation"""
    try:
        print(f"Loading file: {filename}")
        data, fs = sf.read(filename)
        
        # Handle mono/stereo formats
        if len(data.shape) > 1 and data.shape[1] > 1:
            print("Playing stereo audio")
        else:
            print("Playing mono audio")
        
        # Validate and fix audio data
        data = validate_audio(data)
        
        # Normalize if needed
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            print("Normalizing audio...")
            data = data / max_val
        
        # Play audio
        print(f"Playing audio ({len(data)/fs:.2f} seconds at {fs} Hz)")
        sd.play(data, fs)
        sd.wait()
        print("Playback complete.")
        return True
        
    except sf.SoundFileError as e:
        print(f"Error: Could not read audio file. Details: {e}")
        return False
    except sd.PortAudioError as e:
        print(f"Error: Audio playback failed. Details: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUsage: python listen.py <filename.wav>")
        print("\nAvailable audio files:")
        
        # List available audio files in recovered and denoised directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for dir_name in ['recovered', 'denoised']:
            dir_path = os.path.join(current_dir, dir_name)
            if os.path.exists(dir_path):
                wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
                if wav_files:
                    print(f"\n{dir_name} directory:")
                    for i, file in enumerate(wav_files):
                        print(f"  {i+1}. {file}")
        
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)
        
    play_audio(filename)