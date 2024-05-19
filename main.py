# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pyaudio

import io
import numpy as np
import torch
import torchaudio
#import matplotlib
import matplotlib.pylab as plt
import pyaudio
import jupyterplot
torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 8000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def plot_secs():

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    data = []
    voiced_confidences = []

    num_samples = 1536
    print("Started Recording")
    for i in range(0, 20):
        audio_chunk = stream.read(num_samples)

    # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16);

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 8000).item()
        voiced_confidences.append(new_confidence)

    print("Stopped the recording")

    # plot the confidences for the speech
    plt.figure(figsize=(20, 6))
    plt.plot(voiced_confidences)
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_audiodevices():
    p = pyaudio.PyAudio()
    print("Available devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{info['index']}: {info['name']} - Input Channels: {info['maxInputChannels']}")

    p.terminate()


global continue_recording


def stop():
    input("Press Enter to stop the recording:")
    #global continue_recording
    stream.close()
    audio.terminate()
    continue_recording = False


def start_recording():
    from jupyterplot import ProgressPlot
    import threading
    num_samples = 1536
    #audio = pyaudio.PyAudio()
    #stream = audio.open(format=FORMAT,
                        #channels=CHANNELS,
                        #rate=SAMPLE_RATE,
                        #input=True,
                        #frames_per_buffer=CHUNK)
    data = []
    voiced_confidences = []
    continue_recording = True

    #pp = ProgressPlot(plot_names=["Silero VAD"], line_names=["speech probabilities"], x_label="audio chunks")

    stop_listener = threading.Thread(target=stop)
    stop_listener.start()

    countsilence=0
    while continue_recording:
        audio_chunk = stream.read(num_samples)

        # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16);

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 8000).item()
        #voiced_confidences.append(new_confidence)
        if new_confidence > 0.4:
            print("Speech " + str(new_confidence))
            countsilence=0
        else:
            countsilence = countsilence+1
            if countsilence > 10:
                print("Long silence " + str(new_confidence))
                #stop()
            else:
                print("-- -- -- " + str(new_confidence))



        #pp.update(new_confidence)

    #pp.finalize()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm test')
    #print_audiodevices()
    #plot_secs()
    start_recording()


