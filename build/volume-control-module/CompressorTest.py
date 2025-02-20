import os
import math
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

def convert_wav_to_usable_format(input_file, temp_file):
    """
    Convert the input .wav file to mono, 32-bit floating-point format and save it as a temporary file.
    """
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)

    # Convert to mono and 32-bit floating-point format
    audio = audio.set_channels(1)  # Mono
    audio = audio.set_sample_width(4)  # 32-bit floating-point

    # Export the converted audio to a temporary file
    audio.export(temp_file, format="wav")

def linear_gain_to_db(gain):
    """
    Convert a linear gain value to decibels (dB).
    """
    if gain <= 0:
        raise ValueError("Gain must be a positive number.")
    return 20 * math.log10(gain)

def process_audio(input_file, output_path, volumetype, gain=None):
    """
    Process the audio file based on volumetype and apply compression.
    Use pydub's volume control for applying gain.
    """
    # Create a temporary file for the converted audio
    temp_file = "temp_audio.wav"

    # Convert the input file to mono, 32-bit floating-point format
    convert_wav_to_usable_format(input_file, temp_file)

    # Load the converted .wav file
    audio = AudioSegment.from_file(temp_file)

    # Apply volume adjustment based on volumetype
    if volumetype == 0:
        if gain is None:
            raise ValueError("Gain must be provided when volumetype is 0.")
        # Convert the linear gain value to dB
        gain_db = linear_gain_to_db(gain)
        print(f"Applying gain: {gain_db:.2f} dB")
        # Apply the specified gain in dB
        audio = audio.apply_gain(gain_db)
    elif volumetype == 1:
        # Normalize the audio to 0 dBFS (full scale)
        audio = audio.normalize()

    # Apply compression using pydub
    compressed_audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=10, release=100)

    # Export the compressed audio to a new .wav file
    compressed_audio.export(output_path, format="wav")

    # Clean up the temporary file
    os.remove(temp_file)

#Example 
input_file = "input.wav"
output_file = "output.wav"
volumetype = 1  # 0 for fixed gain, 1 for normalization
gain = 10.0  # 2x volume increase (linear gain)

# If volumetype is 0, provide a gain value. If volumetype is 1, gain is automated.
process_audio(input_file, output_file, volumetype, gain)
print(f"Processed audio saved to {output_file}")
