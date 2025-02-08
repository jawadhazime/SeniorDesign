#Script that integrates mic/speaker code with machine learning model code

#################### IMPORTS ############################
import pyaudio
import numpy as np
import sys
import traceback

import torch
import torchaudio
from denoiser import pretrained
import numpy as np
import os
from denoiser.dsp import convert_audio
import time

#################### MIC/SPEAKER INIT ############################

# Configuration
RATE = 48000  # Updated Sampling rate in Hz
CHUNK = 8064  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Number of channels (1 for mono, 2 for stereo)

# Audio device indices based on your device list
INPUT_DEVICE_INDEX = 1    # Plugable USB Audio Device - Input
OUTPUT_DEVICE_INDEX = 1   # Plugable USB Audio Device - Output

#################### MODEL INIT ############################

# Initialize the model
model = pretrained.dns64()

#################### MAIN SCRIPT ############################

def main():
    p = pyaudio.PyAudio()

    try:
        # Open input stream (Microphone)
        stream_input = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=INPUT_DEVICE_INDEX)
        
        # Open output stream (Earpiece)
        stream_output = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            output=True,
                            frames_per_buffer=CHUNK,
                            output_device_index=OUTPUT_DEVICE_INDEX)
        while True:
            try:

                # Read data from microphone as float
                data = np.frombuffer(stream_input.read(CHUNK,exception_on_overflow=False ),dtype=np.float32)
                data = torch.from_numpy(data)

                #PASS THROUGH MODEL
                print(data.dtype)
                # with torch.no_grad():
                #     denoised = model(data[None])[0]
                # print(denoised)
                data = data.numpy()
                print(data.tobytes())
                # Write data to earpiece
                data = stream_input.read(CHUNK, exception_on_overflow=False)
                stream_output.write(data)
                # stream_output.write(data.tobytes())
            except:
                #Add later
                print("exception @ data stream: fkin johnnie walker")
                break

    except:

        #Add later
        print("exception @ stream open: fkin johnnie walker error")
        

    finally:

        stream_input.stop_stream()
        stream_input.close()

        stream_output.stop_stream()
        stream_output.close()
    
    
if __name__ == "__main__":
    main()
        