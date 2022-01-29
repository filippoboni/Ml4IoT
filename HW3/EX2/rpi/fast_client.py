import tensorflow as tf
import os
import pathlib
import numpy as np
import base64
from datetime import datetime
import json
import requests

# CLASSES AND FUNCTIONS------------------------------------------------------------------------------------------------

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_len, frame_step, num_mel_bins=None,
                 num_coefficients=None, lower_freq=None, upper_freq=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_len = frame_len
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        num_spectrogram_bins = frame_len // 2 + 1

        if mfcc:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins,
                                                                                     num_spectrogram_bins,
                                                                                     self.sampling_rate,
                                                                                     self.lower_freq,
                                                                                     self.upper_freq)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        # extract labels
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        # squeeze audio into vector
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        # pad audio if needed
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        # compute audio spectrogram
        stft = tf.signal.stft(signals=audio,
                              frame_length=self.frame_len,
                              frame_step=self.frame_step,
                              fft_length=self.frame_len)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix,
                                       1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        # take only first num_coefficients mfccs
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

# define a function to check how confident is the prediction, setting as lower threshold 40% (????)
def success_checker(output_data, threshold):

    data = np.squeeze(output_data, axis=0)
    data = tf.nn.softmax(data).numpy()
    sorted_idxs = np.argsort(data) #indexes array of the same shape of data

    first = data[sorted_idxs[-1]] #most probable label
    second = data[sorted_idxs[-2]] #2nd most probable label

    if first - second <= threshold:
        return True
    else:
        return False


# Define pre-processing options
MFCC_OPTIONS_DEFAULT = {'frame_len':640, 'frame_step':320, 'mfcc':True,
                'lower_freq':20, 'upper_freq':4000, 'num_mel_bins':40,
                'num_coefficients':10 }


# download data if needed
data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      fname='mini_speech_commands.zip',
      extract=True,
      cache_dir='.', cache_subdir='data')

# load labels list
labels = open('labels.txt').readlines()[0].split()

# lista di test
test_list = []
file = open("kws_test_split.txt")
for line in file:
    test_list.append('.' + line[1:-1])

generator = SignalGenerator(labels, 16000, **MFCC_OPTIONS_DEFAULT)
test_ds = generator.make_dataset(test_list, train=False)

# load model
tflite_interpreter = tf.lite.Interpreter(model_path='./kws_dscnn_True.tflite')
tflite_interpreter.allocate_tensors()
# Input and output tensors
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
tflite_interpreter.invoke()

tot_el = 0
weight = 0
corrects = 0
threshold = 0.4
url = 'http://raspberrypi.local:8080'

for sample, audio_binary, label in test_ds:

    tot_el +=1
    sample = np.expand_dims(sample, axis=0).astype(np.float32)
    tflite_interpreter.set_tensor(input_details[0]['index'], sample)
    tflite_interpreter.invoke()
    output_data = tflite_interpreter.get_tensor(output_details[0]['index'])

    if success_checker(output_data, threshold=threshold):
        # create json audio_binary to pass to service
        audio_bytes = audio_binary.numpy()
        audio_b64 = base64.b64encode(audio_bytes)
        audio_str = audio_b64.decode()
        timestamp = int(datetime.now().timestamp())

        body = {
            "bn" : "raspberrypi.local",
            "bt" : timestamp,
            "e" : [ {"n":"audio", "u":"/", "t":0, "vd": audio_str}]
        }

        weight += len(json.dumps(body))
        r = requests.post(url, body)

        if r.status_code==200:
            r_body = r.json()
            prediction = r_body['prediction']
        else:
            print("Error")
            print(r.text)

    else:
        prediction = np.argmax(output_data)

    if prediction == label:
        corrects +=1

accuracy = corrects/tot_el

print('Accuracy: {} %%'.format(accuracy*100))
print('communication cost: {}'.format(weight/(1024**2)))