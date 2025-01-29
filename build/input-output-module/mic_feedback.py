import pyaudio
import numpy as np
import sys
import traceback

# Configuration
RATE = 48000  # Updated Sampling rate in Hz
CHUNK = 1000  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Number of channels (1 for mono, 2 for stereo)

# Audio device indices based on your device list
INPUT_DEVICE_INDEX = 1    # Plugable USB Audio Device - Input
OUTPUT_DEVICE_INDEX = 1   # Plugable USB Audio Device - Output

def list_audio_devices(p):
    """Function to list all audio devices with their indices."""
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        input_channels = info['maxInputChannels']
        output_channels = info['maxOutputChannels']
        device_type = []
        if input_channels > 0:
            device_type.append('Input')
        if output_channels > 0:
            device_type.append('Output')
        device_type = ' & '.join(device_type) if device_type else 'Inactive'
        print(f"{i}: {info['name']} - {device_type}")

def main():
    p = pyaudio.PyAudio()

    # Uncomment the following line to list all available audio devices
    list_audio_devices(p)
    print("\nAttempting to start audio pass-through...\n")

    try:
        # Open input stream (Microphone)
        stream_input = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK,
                              input_device_index=INPUT_DEVICE_INDEX)
        print(f"Input stream opened: {stream_input}")

        # Open output stream (Earpiece)
        stream_output = p.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               output=True,
                               frames_per_buffer=CHUNK,
                               output_device_index=OUTPUT_DEVICE_INDEX)
        print(f"Output stream opened: {stream_output}")

        print("Audio pass-through started. Press Ctrl+C to stop.")

        while True:
            try:
                # Read data from microphone
                data = stream_input.read(CHUNK, exception_on_overflow=False)
                print(data)

                # Optionally, process the data here (e.g., apply filters)

                # Write data to earpiece
                stream_output.write(data)

            except IOError as e:
                # Handle buffer overflows
                print(f"I/O error({e.errno}): {e.strerror}", file=sys.stderr)
            except KeyboardInterrupt:
                # Handle user interruption
                print("\nAudio pass-through stopped by user.")
                break
            except Exception as e:
                # Catch-all for other exceptions
                print("An unexpected error occurred:", file=sys.stderr)
                traceback.print_exc()

    except Exception as e:
        print("Failed to open audio streams:", file=sys.stderr)
        traceback.print_exc()

    finally:
        # Close streams if they were opened
        try:
            if 'stream_input' in locals():
                stream_input.stop_stream()
                stream_input.close()
                print("Input stream closed.")
        except Exception as e:
            print("Error closing input stream:", file=sys.stderr)
            traceback.print_exc()

        try:
            if 'stream_output' in locals():
                stream_output.stop_stream()
                stream_output.close()
                print("Output stream closed.")
        except Exception as e:
            print("Error closing output stream:", file=sys.stderr)
            traceback.print_exc()

        # Terminate PyAudio
        p.terminate()
        print("PyAudio terminated.")

if __name__ == "__main__":
    main()
