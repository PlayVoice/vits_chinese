import os
import torch
import numpy as np
import argparse
import utils
import random

from bert import TTSProsody
from bert.prosody_tool import is_chinese, pinyin_dict
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch


def get_spec(hps, filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    assert sampling_rate == hps.data.sampling_rate, f"{sampling_rate} is not {hps.data.sampling_rate}"
    audio_norm = audio / hps.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = torch.squeeze(spec, 0)
    return spec


def get_spk_map(path):
    speaker = os.listdir(path)
    speaker.sort()
    speaker_map = {}
    for i in range(len(speaker)):
        speaker_map[speaker[i]] = i
    print(speaker_map)
    return speaker_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", dest="conf", required=True)
    parser.add_argument("--data", dest="data", required=True)
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.data, "waves-16k"))
    assert os.path.exists(os.path.join(args.data, "lables.txt"))

    speaker_map = get_spk_map(os.path.join(args.data, "waves-16k"))
    fout = open(os.path.join(args.data, "speakers.txt"), 'w', encoding='utf-8')
    print(speaker_map, file=fout)
    fout.close()

    wave_path = os.path.join(args.data, "waves-16k")
    bert_path = os.path.join(args.data, "berts")
    spec_path = os.path.join(args.data, "temps")
    os.makedirs(bert_path, exist_ok=True)
    os.makedirs(spec_path, exist_ok=True)

    hps = utils.get_hparams_from_file(args.conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prosody = TTSProsody("./bert", device)

    fo = open(os.path.join(args.data, "lables.txt"), "r+", encoding='utf-8')
    scrips = []
    while (True):
        try:
            message = fo.readline().strip()
            pinyins = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (message == None):
            break
        if (message == ""):
            break
        infosub = message.split(" ")
        fileidx = infosub[0]
        message = infosub[1]
        speaker = fileidx[:7]

        os.makedirs(os.path.join(bert_path, speaker), exist_ok=True)
        os.makedirs(os.path.join(spec_path, speaker), exist_ok=True)

        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)

            pinyins = pinyins.split()
            len_pys = len(pinyins)
            for word in message:
                if is_chinese(word):
                    count_phone.append(2)
                    # if (phone_index >= len_pys):
                    #     print(len_pys)
                    #     print(phone_index)
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                    else:
                        # 资 zi1 本 ben2 发 fa1 现 xian4 了 le5 体 ti3 育 yu4 这 zhe4 块 kuair4 价 jia4 值 zhi2 洼 wa1 地 di4
                        raise IndexError(f'Unkown PinYin: {pinyin}')
                else:
                    count_phone.append(1)
                    phone_items.append('sp')
            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)

        except IndexError as e:
            print(f"{fileidx}\t{message}")
            print('except:', e)
            continue

        text = f'[PAD]{message}[PAD]'
        char_embeds = prosody.get_char_embeds(text)
        char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        char_embeds_path = os.path.join(bert_path, speaker, f"{fileidx}.npy")
        np.save(char_embeds_path, char_embeds, allow_pickle=False)

        wave_file = os.path.join(wave_path, speaker, f"{fileidx}.wav")
        spec_file = os.path.join(spec_path, speaker, f"{fileidx}.spec.pt")
        if not os.path.exists(wave_file):
            continue
        spec = get_spec(hps, wave_file)
        torch.save(spec, spec_file)

        scrips.append(
            f"{speaker_map[speaker]}|{wave_file}|{spec_file}|{char_embeds_path}|{phone_items_str}")

    fo.close()

    os.makedirs('./filelists/', exist_ok=True)
    fout = open(f'./filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[:20]:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/train.txt', 'w', encoding='utf-8')
    tmp = scrips[20:]
    random.shuffle(tmp)
    for item in tmp:
        print(item, file=fout)
    fout.close()
