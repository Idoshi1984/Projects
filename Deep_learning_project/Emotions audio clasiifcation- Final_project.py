'''
Final prpject
Ido Shirat
201116175
classification of audio emotions files
https://www.kaggle.com/uldisvalainis/audio-emotions
'''


## Import packges
# General Tools
import json

import numpy as np
import scipy as sp
import pandas as pd
import glob
from pathlib import Path

# SK Learn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Keras
import tensorflow as tf
import tensorflow_datasets as tfds
#
from tensorflow.keras import Model, Input
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM, RepeatVector, TimeDistributed, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

# Misc
import random
from random import sample
import warnings
from sys import modules
from time import time
import os
from platform import python_version

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# IPython
from IPython.display import Image, display

# Audio
from playsound import playsound
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import librosa
import librosa.display
import time
import glob
import math


#<! Check if GPU is available
#<! See also https://www.tensorflow.org/guide/gpu
#
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Confuguration
# warnings.filterwarnings('ignore')

#
# seedNum = 512
# np.random.seed(seedNum)
# random.seed(seedNum)
#
#
# ## example of audio
# SHOW_EXP = False
# if SHOW_EXP:
#     exp = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Emotions\Angry\03-01-05-01-01-01-01.wav'
#
#     singal, sr = librosa.load(exp, sr=22050) # sr*time= len(singal)
#     sd.play(singal, sr)
#     plt.figure()
#     librosa.display.waveplot(singal,sr=sr)
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.show()
#
#     ##FFT
#     fft = np.fft.fft(singal)
#     magnitude = np.abs(fft)
#     frequency = np.linspace(0,sr,len(magnitude))
#
#     left_frequency = frequency[:int(len(frequency)/2)]
#     left_magnitude = magnitude[:int(len(magnitude)/2)]
#     plt.figure()
#     plt.plot(left_frequency,left_magnitude)
#     plt.xlabel('Frequency')
#     plt.ylabel('Magnitude')
#     plt.show()
#
#     ##STFT
#     n_fft = 2048 # number of samples in a given window
#     hop_length = 512 # window frame
#
#     stft = librosa.core.stft(singal, hop_length= hop_length, n_fft= n_fft)
#     spectrogram = np.abs(stft)
#
#     log_spectrogram = librosa.amplitude_to_db(spectrogram)
#
#     plt.figure()
#     librosa.display.specshow(log_spectrogram, sr= sr, hop_length= hop_length)
#     plt.xlabel('Time')
#     plt.ylabel('Frequency')
#     plt.colorbar()
#     plt.show()
#
#     ## MFCC
#     MFCC = librosa.feature.mfcc(singal, n_fft= n_fft, hop_length= hop_length, n_mfcc= 13)
#     plt.figure()
#     librosa.display.specshow(MFCC, sr= sr, hop_length= hop_length)
#     plt.xlabel('Time')
#     plt.ylabel('MFCC')
#     plt.colorbar()
#     plt.show()
#
#
#


#####
#Func
####

# def save_mfcc(path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments= 3):
#     SAMPLE_RATE = 22050
#     DURATION = 3
#     SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
#
#     # data dict
#     data = {
#         'mapping': [],
#         'mfcc': [],
#         'labels': []
#     }
#
#     num_samples_per_segments = int(SAMPLES_PER_TRACK/num_segments)
#     EXPECTED_NUM_MFCC_VECTORS_PER_SEGMENT = math.ceil(num_samples_per_segments/hop_length)
#
#     for i, (subdir, dirs, files) in enumerate(os.walk(path)):
#
#         path_components = path.split("/")
#         semantic_label = path_components[-1]
#         data['mapping'].append(semantic_label)
#         print(f'/nProccesing {semantic_label}')
#
#         for file in files:
#             try:
#                 #Load librosa array, obtain mfcss, add them to array and then to list.
#                 signal, sr = librosa.load(os.path.join(subdir,file), sr = SAMPLE_RATE)
#
#                 ## no segments
#
#                 mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc,
#                                             hop_length=hop_length, n_fft=n_fft).T
#                 data['mfcc'].append(mfcc.tolist())
#                 data['labels'].append(i-1)
#                 print(f'{os.path.join(subdir, file)}')
#
#                 ## with segments- process segments and get mfcc
#
#                 # for s in range(num_segments):
#                 #     start_sample = num_samples_per_segments * s
#                 #     finish_sample = start_sample + num_samples_per_segments
#                 #
#                 #     mfcc =librosa.feature.mfcc(signal[start_sample:finish_sample], sr =sr, n_mfcc= n_mfcc, hop_length= hop_length, n_fft = n_fft).T
#                 #     if len(mfcc) == EXPECTED_NUM_MFCC_VECTORS_PER_SEGMENT:
#                 #         data['mfcc'].append(mfcc.tolist())
#                 #         data['labels'].append(i-1)
#                 #         print(f'{os.path.join(subdir,file)}, segemnt: {s}')
#             except:
#                 continue
#     with open(json_path, "w") as fp:
#         json.dump(data, fp, indent=4)
#
# def load_data(dataset_path):
#     with open(dataset_path, "r") as fp:
#         data = json.load(fp)
#
#     ## conver list to np
#     inputs = np.array(data["mfcc"])
#     targets = np.array(data["labels"])
#     return  inputs, targets
#
# PATH = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Emotions'
# JSON_PATH = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Proccesed_data_json\data.json'
# SAVE_JSON = False
#
# if SAVE_JSON:
#     save_mfcc(PATH, JSON_PATH)
#
# # load data
# # inputs, targets = load_data(JSON_PATH)
#
# ## train test split
#
# # X_train, X_test, y_train, y_test = train_test_split(
# #     inputs, targets, test_size=0.3, random_state=1)
# #
# # X_train = np.asarray(X_train)
# # X_test = np.asarray(X_test)
# # y_train = np.asarray(y_train)
# # y_test = np.asarray(y_test)
# #
# # X_traincnn = np.expand_dims(X_train, axis=1)
# # X_testcnn = np.expand_dims(X_test, axis=1)
# #
# # y_train = np_utils.to_categorical(y_train,7)
# # y_test = np_utils.to_categorical(y_test,7)
# #
# # ## build architechture
# # model = keras.Sequential([
# #  keras.layers.Flatten(input_shape= (inputs.shape[1], inputs.shape[2])),
# #     keras.layers.Dense(512,activation= 'relu'),
# #     keras.layers.Dense(256,activation= 'relu'),
# #     keras.layers.Dense(64,activation= 'relu'),
# #     keras.layers.Dense(7,activation='softmax')
# # ])
# # ## complie network
# # optimizer = keras.optimizers.Adam(learning_rate= 0.001)
# # model.compile(optimizer= optimizer, loss= 'categorical_crossentropy', metrics= ['accuracy'])
# # model.summary()
# # ## train network
# # model.fit(X_train, y_train,  batch_size=200, epochs=50, verbose=1,validation_split=0.2)
# a=1
#


#############################################
###########################################
# def save_mfccs(audio_path,csv_save_path,n_MFCC,SR,n_fft, hop_length,resample_type):
#
#     i = -2
#     lst = []
#     dict_data = {'mfcc':[], 'labels':[],'mapping':[], 'mel':[]}
#     start_time = time.time()
#     for subdir, dirs, files in os.walk(audio_path):
#         i=i+1
#         print(subdir)
#         print(i)
#         path_components = subdir.split("\\")
#         label = path_components[-1]
#         for file in files:
#             #Load librosa array, obtain mfcss, add them to array and then to list.
#             signal, sample_rate = librosa.load(os.path.join(subdir,file),sr = SR, res_type=resample_type)
#
#             # S = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
#             S = librosa.feature.melspectrogram(signal, sr=sample_rate, n_mels=40, fmax=8000).T
#             mel = librosa.power_to_db(S,ref = np.max)
#
#             mfccs = librosa.feature.mfcc(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_MFCC).T
#             arr = mfccs, i, label
#             dict_data['mfcc'].append(mfccs.tolist())
#             dict_data['labels'].append(i-1)
#             dict_data['mapping'].append(label)
#             dict_data['mel'].append(mel.tolist())
#
#
#             # lst.append(arr) #Here we append the MFCCs to our list.
#
#
#     with open(
#             r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Proccesed_data_json\data_spec.json',
#             "w") as fp:
#         json.dump(dict_data, fp, indent=4)
#     print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
# ##################
#
# def save_mfccs1(audio_path, json_save_path, n_MFCC, SR, n_fft, hop_length, resample_type):
#     noise = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)])
#     time_strech = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5), ])
#     pitch_shift = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=0.7)])
#     shift = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)])
#
#     augmnet_list = [noise,time_strech,pitch_shift,shift]
#
#     i = -2
#     lst = []
#     dict_data = {'mfcc': [], 'labels': [], 'mapping': []}
#     for subdir, dirs, files in os.walk(audio_path):
#         i = i + 1
#         print(subdir)
#         print(i)
#         path_components = subdir.split("\\")
#         label = path_components[-1]
#         for file in files:
#             # Load librosa array, obtain mfcss, add them to array and then to list.
#             signal, sample_rate = librosa.load(os.path.join(subdir, file), sr=SR, res_type=resample_type)
#             mfccs = librosa.power_to_db(librosa.feature.mfcc(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_MFCC).T,ref=np.max)
#             dict_data['mfcc'].append(mfccs.tolist())
#             dict_data['labels'].append(i - 1)
#             dict_data['mapping'].append(label)
#
#             for augment in augmnet_list:
#                 augmented_file = augment(samples=signal, sample_rate=sample_rate)
#                 mfccs = librosa.feature.mfcc(augmented_file, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_MFCC).T
#                 dict_data['mfcc'].append(mfccs.tolist())
#                 dict_data['labels'].append(i - 1)
#                 dict_data['mapping'].append(label)
#
#     with open(json_save_path, "w") as fp:
#         json.dump(dict_data, fp, indent=4)
#     a=1
#
#
# def save_mfccs2(audio_path, json_save_path, n_MFCC, SR, n_fft, hop_length, resample_type):
#
#     i = -2
#     X_np = np.zeros([12798,500000])
#     y_np = np.empty([12798,1])
#     label_np = np.empty([12798,1])
#     count = 0
#     for subdir, dirs, files in os.walk(audio_path):
#         i = i + 1
#         print(subdir)
#         print(i)
#         path_components = subdir.split("\\")
#         label = path_components[-1]
#         for file in files:
#             # Load librosa array, obtain mfcss, add them to array and then to list.
#             signal, sample_rate = librosa.load(os.path.join(subdir, file), sr=SR, res_type=resample_type)
#             X_np[count,0:len(signal)] = signal
#             y_np[count, :] = i
#             label_np[count,:] = label
#
#     np.save(r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\X_np.npy', X_np)
#     np.save(r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\y_np.npy', y_np)
#     np.save(r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\label_np.npy', label_np)
#
#     a=1
#
#
# def load_csv(csv_save_path):
#     data = pd.read_csv(csv_save_path, converters={"mfcc": lambda x: x.strip("[]").split(", ")})
#     data['mfcc'] = data['mfcc'].explode().str.split()
#     X = pd.Series()
#     for  row in range(len(data)):
#        X.loc[row]= [float(i) for i in data.loc[row,'mfcc']]
#     X = tuple(X)
#     y = data['label_num']
#     labels = data['label_str']
#     X = np.asarray(X)
#     y = np.asarray(y)
#     return X, y, labels
#
# def load_json(json_path):
#     with open(json_path, "r") as fp:
#         data = json.load(fp)
#         X = np.array(data['mfcc'])
#         y = np.array(data['labels'])
#         X_extra = np.empty ((12798,500,40))
#         for i in range(len(X)):
#             extra =500-np.array(X[i]).shape[0]
#             X_extra[i,:,:] = np.pad(np.array(X[i]),pad_width=((extra,0),(0,0)), mode='constant', constant_values=0)
#         return X_extra, y
#
#
#
#
# json_path = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Proccesed_data_json\data.json'
# audio_path = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Emotions'
# n_MFCC = 40
# resample_type = 'kaiser_fast' # kaiser_best, kaiser_fast
# SR = 22050
# n_fft = 2048
# hop_length = 512
# csv_save_path = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Proccesed_data_csv\data.csv'
# save_mfcc = True
# if save_mfcc:
#     save_mfccs2(audio_path,csv_save_path,n_MFCC,SR,n_fft, hop_length,resample_type)
# # X,y,z = zip(*lst)
#
#
#
#
# ################
# # X, y, labels = load_csv(csv_save_path)
#
# X, y = load_json(json_path)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=1)
#
# X_traincnn = np.expand_dims(X_train, axis=3)
# X_testcnn = np.expand_dims(X_test, axis=3)
#
# y_train = np_utils.to_categorical(y_train,7)
# y_test = np_utils.to_categorical(y_test,7)
# # baseModel = keras.applications.ResNet50(
# #     weights = "VGG-19",  #<! Load weights pre-trained on ImageNet. One could use a file path!
# #     input_shape = (None, None, 1),
# #     include_top = False, #<! Last layer (The classifier for 1000 classes)
# # )  #<! Do not include the ImageNet classifier at the top.
# #
# # baseModel.trainable = False #<! Freeze the base model
# # baseModel.summary()
#
# #optimzers
# initLr = 0.05
# decay_s = 150
# decay_r = 0.96
# staircase = True
# numEpochs = 500 #<! Don't change
# batchSize = 250
# lrSchedule = keras.optimizers.schedules.ExponentialDecay(initLr, decay_steps = decay_s, decay_rate = decay_r, staircase = staircase)
# hOpt = keras.optimizers.Adam(learning_rate = lrSchedule, name = "Adam")
#
# inputlayer = Input(shape=(X_testcnn.shape[1],X_testcnn.shape[2], X_testcnn.shape[3]))
# x = Flatten()(inputlayer)
# x = Dense(100, activation='relu')(x)
# x = Dropout(0.25)(x)
# x = Dense(100, activation='relu')(x)
# x = Dropout(0.25)(x)
# outputs = Dense(7, activation='softmax')(x)
# model = Model(inputs=inputlayer, outputs=outputs)
# model.compile(loss='categorical_crossentropy', optimizer=hOpt, metrics=['accuracy'])
#
# model.fit(X_traincnn, y_train, batch_size=batchSize, epochs=numEpochs, verbose=1, validation_split=0.2)
#
# score = model.evaluate(X_testcnn, y_test, verbose=0)
# print('\nScore: ', score)
# ##################
#
#
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation="relu", input_shape=(X.shape[1],X.shape[2], X.shape[3])))
# model.add(MaxPooling2D(pool_size= (2,2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(300, activation= 'relu'))
# model.add(Dense(7, activation = 'softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# model.fit(X_traincnn, y_train, batch_size=batchSize, epochs=numEpochs, verbose=1, validation_split=0.2)
# score = model.evaluate(X_testcnn, y_test, verbose=0)
# print('\nScore: ', score)
# a=1


#####################################################################################################

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    print(path)
    SAMPLING_RATE = 22050
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels
DATASET_AUDIO_PATH = r'C:\Users\idos\Desktop\Ido\Data_Science\Naya_course\Final_project\Audio_files\Emotions'
class_names = os.listdir(DATASET_AUDIO_PATH)
print("Our class names: {}".format(class_names,))

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name,))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Shuffle
SHUFFLE_SEED = 123
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

VALID_SPLIT = 0.2
# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

BATCH_SIZE = 100
# Create 2 datasets, one for training and the other for validation
train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)



# Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

SAMPLING_RATE = 22050
model = build_model((SAMPLING_RATE // 2, 1), len(class_names))




model = build_model((SAMPLING_RATE // 2, 1), len(class_names))

model.summary()

# Compile the model using Adam's default learning rate
model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Add callbacks:
# 'EarlyStopping' to stop training when the model is not enhancing anymore
# 'ModelCheckPoint' to always keep the model that has the best val_accuracy
model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)

EPOCHS = 5
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)
a=1