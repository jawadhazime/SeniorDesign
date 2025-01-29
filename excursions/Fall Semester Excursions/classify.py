import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
import matplotlib.pyplot as plt
import sys
import time
import psutil
import os
from scipy.io import wavfile
import io
import kagglehub

#============================================================INPUTS==============================================#
wav_file_name="noisy.wav" #original audio
wav_file_name2="clean.wav" #audio to compare against

#============================================================INITIALIZATION==============================================#
# Read the WAV file
sample_rate, wav_data = wavfile.read(wav_file_name)

# Show some basic information about the audio.
duration = len(wav_data) / sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Normalize the waveform
waveform = wav_data / tf.int16.max

# Load the pre-trained YAMNet model from local directory
print("Downloading latest YAMNet model...")
# Download latest version
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")

print("Path to model files:", path)
model = tf.saved_model.load(path)
print("YAMNet model loaded successfully.")


# Convert stereo to mono if necessary
if len(waveform.shape) > 1 and waveform.shape[1] > 1:
    print("Converting stereo audio to mono...")
    waveform = np.mean(waveform, axis=1)
else:
    waveform = waveform.squeeze()

#============================================================RUNNING THE MODEL==============================================#
# Run the model
print("Running the YAMNet model...")
scores, embeddings, spectrogram = model(waveform)

#============================================================SYNTHESIZE RESULTS==============================================#
# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
print(class_names[scores.numpy().mean(axis=0).argmax()])  # Should print 'Silence'.

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform
    else:
        return original_sample_rate, waveform

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
mean_scores = np.mean(scores_np, axis=0)
inferred_class = class_names[mean_scores.argmax()]
print(f'The main sound is: {inferred_class}')

# Map frames to timestamps
num_frames = scores_np.shape[0]
time_per_frame = duration / num_frames
timestamps = np.arange(num_frames) * time_per_frame

# Divide the length of the audio file by 10 to have 10 equal sample timestamps
num_segments = 10
frames_per_segment = num_frames // num_segments
segment_scores_list = np.array_split(scores_np, num_segments)

# Define speech-related class indices
speech_class_keywords = ['speech', 'conversation', 'narration', 'monologue', 'talking',
                         'dialogue', 'babbling', 'whispering', 'shout', 'bellow', 'yell']

speech_class_indices = []
for idx, name in enumerate(class_names):
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in speech_class_keywords):
        speech_class_indices.append(idx)

# Define noise class indices as all other indices
noise_class_indices = list(set(range(len(class_names))) - set(speech_class_indices))

# Process each segment
print("\nProcessing each segment:")
for i, segment_scores in enumerate(segment_scores_list):
    mean_segment_scores = np.mean(segment_scores, axis=0)

    speech_prob_sum = np.sum(mean_segment_scores[speech_class_indices])
    noise_prob_sum = np.sum(mean_segment_scores[noise_class_indices])
    total_prob_sum = speech_prob_sum + noise_prob_sum
    print(f"Speech probability sum before: {speech_prob_sum}")
    print(f"Noise probability sum before: {noise_prob_sum}")
    print(f"Total probability sum before: {total_prob_sum}")
    print("\n\n")
    # if total_prob_sum == 0:
    #     noise_significance = 0
    # else:
    #     # Noise significance rated from 0 to 10
    #     noise_significance = 10 * (noise_prob_sum / total_prob_sum)

    # # Determine if there is noise other than human speech
    # noise_present = noise_prob_sum > 0.1  # You can adjust the threshold as needed

    # # Get the top 3 noise probabilities
    # noise_probs = mean_segment_scores[noise_class_indices]
    # top_3_noise_indices_in_noise = noise_probs.argsort()[-3:][::-1]
    # top_3_noise_indices = [noise_class_indices[idx] for idx in top_3_noise_indices_in_noise]
    # top_3_noise_classes = [class_names[idx] for idx in top_3_noise_indices]
    # top_3_noise_probs = [mean_segment_scores[idx] for idx in top_3_noise_indices]

    # # Output the results for this segment
    # segment_start_time = i * (duration / num_segments)
    # segment_end_time = (i + 1) * (duration / num_segments)
    # print(f"\nSegment {i+1} ({segment_start_time:.2f}s - {segment_end_time:.2f}s):")
    # print(f"  Noise significance: {noise_significance:.2f} out of 10")
    # if noise_present:
    #     print(f"  Noise present other than human speech.")
    #     print(f"  Top 3 noise classes:")
    #     for cls, prob in zip(top_3_noise_classes, top_3_noise_probs):
    #         print(f"    {cls}: {prob:.4f}")
    # else:
    #     print("  No significant noise present other than human speech.")

#============================================================PLOTTING==============================================#
# # Plotting (optional)
# plt.figure(figsize=(10, 6))

# # Plot the waveform.
# plt.subplot(3, 1, 1)
# plt.plot(waveform)
# plt.xlim([0, len(waveform)])
# plt.title('Waveform')

# # Plot the log-mel spectrogram (returned by the model).
# plt.subplot(3, 1, 2)
# plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
# plt.title('Spectrogram')

# # Plot and label the model output scores for the top-scoring classes.
# mean_scores = np.mean(scores, axis=0)
# top_n = 10
# top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
# plt.subplot(3, 1, 3)
# plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
# plt.title('Top Class Scores Over Time')

# # Adjust x-axis for time
# plt.xlabel('Time (frames)')
# plt.ylabel('Classes')
# yticks = range(0, top_n, 1)
# plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
# plt.tight_layout()
# plt.show()

#============================================================INITIALIZATION (COMPARISON)==============================================#
# Read the WAV file
sample_rate2, wav_data2 = wavfile.read(wav_file_name2)

# Show some basic information about the audio.
duration2 = len(wav_data2) / sample_rate2
print(f'Sample rate: {sample_rate2} Hz')
print(f'Total duration: {duration2:.2f}s')
print(f'Size of the input: {len(wav_data2)}')

# Normalize the waveform
waveform2 = wav_data2 / tf.int16.max

# Convert stereo to mono if necessary
if len(waveform2.shape) > 1 and waveform2.shape[1] > 1:
    print("Converting stereo audio to mono...")
    waveform2 = np.mean(waveform2, axis=1)
else:
    waveform2 = waveform2.squeeze()

#============================================================RUNNING THE MODEL (COMPARISON)==============================================#
# Run the model
print("Running the YAMNet model...")
scores2, embeddings2, spectrogram2 = model(waveform2)


#============================================================SYNTHESIZE RESULTS (COMPARISON)==============================================#
# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
print(class_names[scores.numpy().mean(axis=0).argmax()])  # Should print 'Silence'.

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform
    else:
        return original_sample_rate, waveform

scores_np = scores2.numpy()
spectrogram_np = spectrogram2.numpy()
mean_scores = np.mean(scores_np, axis=0)
inferred_class = class_names[mean_scores.argmax()]
print(f'The main sound is: {inferred_class}')

# Map frames to timestamps
num_frames = scores_np.shape[0]
time_per_frame = duration / num_frames
timestamps = np.arange(num_frames) * time_per_frame

# Divide the length of the audio file by 10 to have 10 equal sample timestamps
num_segments = 10
frames_per_segment = num_frames // num_segments
segment_scores_list = np.array_split(scores_np, num_segments)

# Define speech-related class indices
speech_class_keywords = ['speech', 'conversation', 'narration', 'monologue', 'talking',
                         'dialogue', 'babbling', 'whispering', 'shout', 'bellow', 'yell']

speech_class_indices = []
for idx, name in enumerate(class_names):
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in speech_class_keywords):
        speech_class_indices.append(idx)

# Define noise class indices as all other indices
noise_class_indices = list(set(range(len(class_names))) - set(speech_class_indices))

# Process each segment
print("\nProcessing each segment:")
for i, segment_scores in enumerate(segment_scores_list):
    mean_segment_scores = np.mean(segment_scores, axis=0)

    speech_prob_sum2 = np.sum(mean_segment_scores[speech_class_indices])
    noise_prob_sum2 = np.sum(mean_segment_scores[noise_class_indices])
    total_prob_sum2 = speech_prob_sum2 + noise_prob_sum2
    print(f"Speech probability sum after: {speech_prob_sum2}")
    print(f"Noise probability sum after: {noise_prob_sum2}")
    print(f"Total probability sum after: {total_prob_sum2}")
    print("\n\n")

#============================================================COMPARING RESULTS==============================================#


#============================================================RPI METRICS==============================================#
# # Collect performance metrics after processing
# processing_time = time.time() - start_time
# mem_usage_after = process.memory_info().rss / (1024 * 1024)  # in MB
# cpu_percent = psutil.cpu_percent(interval=1)
# temp_after = get_rpi_temperature()

# # Display performance metrics
# print("\nPerformance Metrics:")
# print(f"Processing time: {processing_time:.2f} seconds")
# print(f"Memory usage before: {mem_usage_before:.2f} MB")
# print(f"Memory usage after: {mem_usage_after:.2f} MB")
# print(f"CPU usage: {cpu_percent}%")
# if temp_before is not None and temp_after is not None:
#     print(f"Raspberry Pi temperature before: {temp_before:.2f}°C")
#     print(f"Raspberry Pi temperature after: {temp_after:.2f}°C")
# else:
#     print("Could not read Raspberry Pi temperature.")


# # Function to get Raspberry Pi temperature
# def get_rpi_temperature():
#     try:
#         temp_str = os.popen("vcgencmd measure_temp").readline()
#         temp = float(temp_str.replace("temp=", "").replace("'C\n", ""))
#         return temp
#     except Exception:
#         return None

# Start performance metrics
# start_time = time.time()
# process = psutil.Process(os.getpid())
# mem_usage_before = process.memory_info().rss / (1024 * 1024)  # in MB
#temp_before = get_rpi_temperature()

# Accept any sound file as input
# if len(sys.argv) > 1:
#     wav_file_name = sys.argv[1]
# else:
#     wav_file_name = "noisy.wav"
    
    #enter above path to wav file
