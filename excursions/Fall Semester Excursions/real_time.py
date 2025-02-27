# -*- coding: utf-8 -*-
"""Denoiser Example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Too3cnMpyKaLQ0vPwDw7jUx0Y3eXm2IA
"""

# !pip install -U denoiser

from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
import soundfile as sf
import numpy as np
import os
from denoiser.dsp import convert_audio
import time
import sounddevice as sd
import cProfile
import pstats

# Initialize the model
model = pretrained.dns64()

# Load the audio file
waveform, sample_rate = torchaudio.load('noisy1.wav', normalize=True)

# 1 second chunk size in samples
chunk_length = sample_rate  # 1 second of audio

# Function to save a chunk to a temporary .wav file
def save_chunk_to_wav(chunk, chunk_index, sample_rate):
    filename = f"temp_chunk_{chunk_index}.wav"
    # Convert the chunk to numpy array (required by soundfile)
    chunk_np = chunk.numpy().T  # Transpose for correct shape (samples, channels)
    # Write to WAV file
    sf.write(filename, chunk_np, sample_rate)
    return filename

# Function to process the chunk and feed it to the model
def process_chunk_with_model(wav_file):
    print("Model started")
    wav, sr = torchaudio.load(wav_file)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    print("Model finished")
    return denoised.numpy().T  # Return the processed waveform (or model output)

# Function to stream the audio, write chunks to files, and pass them to the model
def stream_and_process_audio(waveform, sample_rate, chunk_length):
    total_samples = waveform.size(1)
    chunk_index = 0

    for start in range(0, total_samples, chunk_length):
        end = min(start + chunk_length, total_samples)
        chunk = waveform[:, start:end]  # Extract the chunk

        # Save the chunk to a temporary .wav file
        print(sample_rate)
        input_filename = save_chunk_to_wav(chunk, chunk_index, sample_rate)
        
        start_time = time.time()
        # Stream the chunk directly to the speakers
        chunk_np = chunk.numpy().T
        print('\nPlaying before..')
        print(chunk_np)
        print('Putting chunk through model..')
        # Process the chunk by passing the wav file to the model
        processed_waveform = process_chunk_with_model(input_filename)
        
        print('\nPlaying after..')
        sd.play(processed_waveform, samplerate=16000)
        sd.wait()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        print('Chunk processing latency:', round(elapsed_time, 2), 'seconds')
        # Delete the temporary file after processing (to keep memory usage low)
        os.remove(input_filename)
        #os.remove(output_filename)

        chunk_index += 1
        
        time.sleep(2)

if __name__ == "__main__":
    with cProfile.Profile() as profile: 
        stream_and_process_audio(waveform, sample_rate, chunk_length)
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("results.prof")
