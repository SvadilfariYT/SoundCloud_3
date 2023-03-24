import os
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
import soundfile

def load_audio(file_path, clip_duration=3):
    # print the filename
    print("Processing file:", file_path)
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

    # Convert the stereo audio to mono if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=0)
    
    # Resample the audio clip to the target sample rate
    resampled_audio_clip = librosa.resample(audio_data, sample_rate, sample_rate/7)

    # Convert the resampled audio clip to a waveform tensor
    waveform = tf.convert_to_tensor(resampled_audio_clip, dtype=tf.float32)

    print("Done!")
    return waveform, sample_rate

def load_audio_files_by_dir(path: str, label:str):
    dataset = []
    # This code uses the pathlib module to get a sorted list of the file paths 
    # for all .wav audio files in a specific directory as a list of strings. 
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        # Access all information contained in the file_path and store it
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split("_nohash_")
        utterance_number = int(utterance_number)
    
        # Load audio
        waveform, sample_rate = load_audio(file_path)
        dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])
        
    return dataset


def load_audio_files_by_type(file_paths, label):
    dataset = []

    for i,file_path in enumerate(file_paths):
        path, filename = os.path.split(file_path)
        # Load audio (will get splitted into subclips)
        waveform, sample_rate = load_audio(file_path)
        
        dataset.append([waveform, sample_rate, label, file_path])

    return dataset


def load_audio_files_by_csv(csv_file_path):
    # Load the CSV file into a pandas DataFrame while skipping blank lines
    df = pd.read_csv(csv_file_path, skip_blank_lines=True, encoding='latin-1', sep=",")
    print(df.columns)

    # Create an empty dictionary to hold the filenames for each type
    file_dict = {}
    not_found_counter = 0
    no_type_counter = 0

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the filename and type for the current row
        input_filepath = "./data/AudioSamples_16bit/" + str(row['filename']) + ".WAV"
        output_filepath = "./data/AudioSamples/Processed"
        file_type = row['type']

        # Skip if no Filename is given
        if pd.isna(row['filename']):
            continue
        
        # Skip if no Type is specified
        if pd.isna(file_type):
            print(f"No type specified for file {input_filepath}")
            no_type_counter += 1
            continue

        # Check if the file exists
        if not os.path.exists(input_filepath):
            print("File " + input_filepath + " does not exist!")
            not_found_counter += 1
            continue

        # Call the slice_audio function to process the input file
        clip_length = 3000 #3000ms (3s) long clips
        overlap = 1500 #1500 (1,5s) overlap
        saved_files = slice_audio(input_filepath, output_filepath, clip_length, overlap)

        # Check if the type already exists in the dictionary
        if file_type in file_dict:
            # If the type exists, add the saved file paths to the list of filenames for that type
            file_dict[file_type].extend(saved_files)
        else:
            # If the type does not exist, create a new key-value pair with the type and the list of saved file paths
            file_dict[file_type] = saved_files

    # Print more Info about the loading process
    if (not_found_counter > 0):
        print(f"WARNING: {not_found_counter} in the CSV-file specified files have not been loaded (not found)")
    if (no_type_counter > 0):
        print(f"WARNING: {no_type_counter} in the CSV-file specified files are missing the 'type'-parameter")

    # Return the resulting dictionary
    return file_dict

def slice_audio(input_filepath, output_filepath, clip_length, overlap):
    print(f"Slicing file: {input_filepath}")
    audio = AudioSegment.from_wav(input_filepath)
    start = 0
    saved_files = []


    while start + clip_length <= len(audio):
        end = start + clip_length
        clip = audio[start:end]

        output_file_name = generate_output_filepath(input_filepath, start, end)

        clip.export(output_file_name, format="wav")
        saved_files.append(output_file_name)
        start += clip_length - overlap

    return saved_files

def generate_output_filepath(input_filepath, start, end):
    input_dir, input_file = os.path.split(input_filepath)
    input_name, input_ext = os.path.splitext(input_file)
    output_dir = os.path.join(input_dir, input_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f"{input_name}_start_{start}_end_{end}{input_ext}"
    output_filepath = os.path.join(output_dir, output_file)
    return output_filepath


def get_spectrogram(waveform):
    ## This funciton has some code from https://www.tensorflow.org/tutorials/audio/simple_audio
    frame_length = 256
    frame_step = 145

    waveform = tf.cast(waveform, tf.float32)

    # Option 1: Use tfio to get the spectrogram
    spect = tfio.audio.spectrogram(input=waveform, nfft=frame_length, window=frame_length, stride=frame_step)

    # Option 2: Use tf.signal processing to get the Short-time Fourier transform (stft)
    spectrogram = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)

    return spectrogram, spect


def create_images(dataset, label_dir, train_percentage):
    # make directory
    test_directory = f'./data/test/{label_dir}/'
    train_directory = f'./data/train/{label_dir}/'

    os.makedirs(test_directory, mode=0o777, exist_ok=True)
    os.makedirs(train_directory, mode=0o777, exist_ok=True)
    
    for i, data in enumerate(dataset):
        # Retrieve the waveform
        waveform = data[0]
        # Create the spectrogram given the waveform
        spectrogram, spect = get_spectrogram(waveform)
        # Retrieve the original filename (without the path leading to it)
        original_filename = os.path.basename(data[3])

        spectrogram_file_name = f'spec_{original_filename}_img{i}.png'

        # Split test and train images by train_percentage
        if i % (train_percentage/10) == 0:
            # Train Data
            plt.imsave(os.path.join(test_directory, spectrogram_file_name), spect.numpy(), cmap='gray')
        else:
            # Test Data
            plt.imsave(os.path.join(train_directory, spectrogram_file_name), spect.numpy(), cmap='gray')


def create_trainData():
    trainset_speechcommands_yes = load_audio_files_by_dir('./data/yes', 'yes')
    trainset_speechcommands_no = load_audio_files_by_dir('./data/no', 'no')
    trainset_speechcommands_dog = load_audio_files_by_dir('./data/dog', 'dog')

    print(f'Length of yes dataset: {len(trainset_speechcommands_yes)}')
    print(f'Length of no dataset: {len(trainset_speechcommands_no)}')
    print(f'Length of dog dataset: {len(trainset_speechcommands_dog)}')

    # train_percentage = 30%
    create_images(trainset_speechcommands_yes, 'yes', 30)
    create_images(trainset_speechcommands_no, 'no', 30)
    create_images(trainset_speechcommands_dog, 'dog', 30)

    print("Done!")

def create_trainData_csv(csv_file_path):
    data = load_audio_files_by_csv(csv_file_path)

    trainset_bus = load_audio_files_by_type(data["Bus"], "Bus")
    trainset_trains_overground = load_audio_files_by_type(data["Train (Overground)"], "Train (Overground)")
    trainset_skateboard = load_audio_files_by_type(data["Skateboard"], "Skateboard")
    trainset_car = load_audio_files_by_type(data["Car"], "Car")

    print(f'Length of BUS dataset: {len(trainset_bus)}')
    print(f'Length of TRAIN(OVERGROUND) dataset: {len(trainset_trains_overground)}')
    print(f'Length of SKATEBOARD dataset: {len(trainset_skateboard)}')
    print(f'Length of CAR dataset: {len(trainset_car)}')

    train_percentage = 30 #(%) - Percentage used to create a 
    create_images(trainset_bus, 'bus', train_percentage)
    create_images(trainset_trains_overground, 'train_overground', train_percentage)
    create_images(trainset_skateboard, 'skateboard', train_percentage)
    create_images(trainset_car, 'car', train_percentage)

def convert_all_files_in_directory_to_16bit(directory):
    for file in os.listdir(directory):
        if file.endswith('.WAV'):
            nameSolo = file.rsplit('.', 1)[0]
            file_path = os.path.join(directory, file)
            print(file_path)
            with soundfile.SoundFile(file_path) as f:
                print(f'converting {file} to 16-bit')
                data, samplerate = f.read(frames=-1, dtype='int16')
                if f.channels == 2:
                    # Convert stereo to mono
                    data = np.mean(data, axis=1)
                soundfile.write(file_path, data, samplerate, subtype='PCM_16')

