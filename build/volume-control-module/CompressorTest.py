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

def calculate_average_loudness(audio):
    """
    Calculate the average loudness of the audio in dBFS.
    """
    return audio.dBFS

def process_audio(input_file, output_path, volumetype, gain=None):
    """
    Process the audio file based on volumetype.
    - If volumetype = 0, apply gain only.
    - If volumetype = 1, apply dynamic compression to achieve an average loudness of -20 dBFS.
    """
    # Create a temporary file for the converted audio
    temp_file = "temp_audio.wav"

    # Convert the input file to mono, 32-bit floating-point format
    convert_wav_to_usable_format(input_file, temp_file)

    # Load the converted .wav file
    audio = AudioSegment.from_file(temp_file)

    # Apply processing based on volumetype
    if volumetype == 0:
        if gain is None:
            raise ValueError("Gain must be provided when volumetype is 0.")
        # Convert the linear gain value to dB
        gain_db = linear_gain_to_db(gain)
        print(f"Applying gain: {gain_db:.2f} dB")
        # Apply the specified gain in dB
        audio = audio.apply_gain(gain_db)
    elif volumetype == 1:
        # Calculate the average loudness of the audio
        average_loudness = calculate_average_loudness(audio)
        print(f"Average loudness: {average_loudness:.2f} dBFS")

        # Set the target loudness to -20 dBFS
        target_loudness = -20.0

        # Calculate the difference between the average loudness and the target loudness
        loudness_diff = average_loudness - target_loudness

        # Adjust the compression threshold dynamically
        # The threshold is set to the target loudness (-20 dBFS) plus the loudness difference
        threshold = target_loudness + loudness_diff
        print(f"Dynamic compression threshold: {threshold:.2f} dBFS")

        # Apply dynamic compression using pydub
        print("Applying dynamic compression...")
        audio = compress_dynamic_range(audio, threshold=threshold, ratio=8.0, attack=15, release=80)

    # Export the processed audio to a new .wav file
    audio.export(output_path, format="wav")

    # Clean up the temporary file
    os.remove(temp_file)

# Example usage
input_file = "input1.wav"
output_file = "outputjawad.wav"
volumetype = 1  # 0 for gain only, 1 for dynamic compression only
gain = 2.0  # Only used when volumetype = 0

# If volumetype is 0, apply gain only. If volumetype is 1, apply dynamic compression only.
process_audio(input_file, output_file, volumetype, gain)
print(f"Processed audio saved to {output_file}")
