import pyaudio
import numpy as np
import keras
import librosa
import librosa.display
import os
import glob
import scipy
import pyaudio
import numpy as np
import time
import soundfile as sf
import matplotlib.pyplot as plt

from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)

windowLength = 255
fftLength = 255
hop_length = 63
frame_length = 8064
chunk = 8064
rate = 22050
#noisy_file = 'test_audio2.wav'
noisy_file = 'noisy1.wav'
clean_file = 'hefaudioOut_v2.wav'
MODEL_NAME = "VOX_MODEL"  # Replace with your model name
HEF_FILE = 'New_FFT_Vox_Model_HEF.hef'

if __name__ == "__main__":
    # MODEL INPUT and OUTPUT
    target = VDevice()
    hef = HEF(HEF_FILE)
    # Configure network groups
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    input_vstreams_info = hef.get_input_vstream_infos()[0]
    output_vstreams_info = hef.get_output_vstream_infos()[0]
    #Creating the audio chunks
    noisy_chunks = []
    noisy_chunks_spect = []
    clean_audio = []
    noisy_sample, noisy_sample_rate = librosa.load(noisy_file)
    print(noisy_sample.shape)
    for j in range(chunk, len(noisy_sample), chunk):
        k = j - chunk
        noisy_chunks.append(noisy_sample[k:j])
    noisy_chunks = np.array(noisy_chunks)
    print(noisy_chunks[1].shape)
    print(len(noisy_chunks))

    for i in range(len(noisy_chunks)): 
        data_stft = librosa.stft(noisy_chunks[i], n_fft=fftLength, hop_length=hop_length)
        chunk_stft_mag, chunkstft_phase =librosa.magphase(data_stft)
        data_stft_mag_db = librosa.amplitude_to_db(chunk_stft_mag, ref=np.max)
        chunk_stft_mag_db_scaled = (data_stft_mag_db+80)/80
        #print("this is the reka shape")
        #print(chunk_stft_mag_db_scaled.shape) # 128,128
        chunk_stft_mag_db_scaled = np.reshape(chunk_stft_mag_db_scaled,(1,chunk_stft_mag_db_scaled.shape[0],chunk_stft_mag_db_scaled.shape[1],1))
        data_stft_mag_db = np.reshape(data_stft_mag_db,(1,data_stft_mag_db.shape[0],data_stft_mag_db.shape[1],1))
        #print(chunk_stft_mag_db_scaled.shape) #1,128,128,1

        with InferVStreams(network_group,input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_data = {input_vstreams_info.name: chunk_stft_mag_db_scaled }
            with network_group.activate(network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
                    #print('Stream output shape is {}'.format(infer_results[output_vstreams_info.name].shape)) #1,128,128,1  
        result_arr = infer_results.get('New_FFT_Vox_Model/conv24')  #1,128,128,1      
        predicted_clean = np.reshape(result_arr, (result_arr.shape[1], result_arr.shape[2]))
        fig2, ax2 = plt.subplots()
        img2 =librosa.display.specshow(predicted_clean, sr=rate, hop_length=hop_length,
                             cmap='viridis', x_axis=None, y_axis=None)
        fig2.colorbar(img2, ax=ax2)
        predicted_mag_db_unscaled = (predicted_clean * 80) - 80
        predicted_mag = librosa.db_to_amplitude(predicted_mag_db_unscaled, ref=np.max(chunk_stft_mag))
        predicted_stft = predicted_mag * chunkstft_phase
        reconstructed_audio =librosa.istft(predicted_stft, hop_length=63, length=8064)
        clean_audio.append(reconstructed_audio)
    clean_audio = np.concatenate(clean_audio, axis=0)
    sf.write(clean_file, clean_audio, samplerate=rate)