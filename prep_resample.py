import os
import librosa
import argparse
import numpy as np

from tqdm import tqdm
from scipy.io import wavfile


def resample_wave(wav_in, wav_out):
    wav, _ = librosa.load(wav_in, sr=24000)
    wav = librosa.resample(wav, orig_sr=24000, target_sr=16000)
    if (len(wav) > 48000):  # 3S
        wav = wav / np.abs(wav).max() * 0.6
        wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
        wavfile.write(wav_out, 16000, wav.astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter path ...'
    parser.add_argument("--wav", dest="wav")
    parser.add_argument("--out", dest="out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out

    todoList = []
    for spks in os.listdir(wavPath):
        if not os.path.isdir(f"./{wavPath}/{spks}"):
            continue
        os.makedirs(f"./{outPath}/{spks}", exist_ok=True)
        for file in os.listdir(f"./{wavPath}/{spks}"):
            if file.endswith(".wav"):
                cell = (f"{wavPath}/{spks}/{file}", f"{outPath}/{spks}/{file}")
                todoList.append(cell)

    for item in tqdm(todoList, desc="normalize waves"):
        resample_wave(item[0], item[1])
