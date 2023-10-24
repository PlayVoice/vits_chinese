import os
import torch
import numpy as np
import argparse
import utils

from bert import TTSProsody
from bert.prosody_tool import is_chinese, pinyin_dict
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch


os.makedirs("./data/waves", exist_ok=True)
os.makedirs("./data/berts", exist_ok=True)
os.makedirs("./data/temps", exist_ok=True)


def log(info: str):
    with open(f'./data/prepare.log', "a", encoding='utf-8') as flog:
        print(info, file=flog)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/bert_vits.json",
        help="JSON file for configuration",
    )
    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    prosody = TTSProsody("./bert", device)

    fo = open(f"./data/000001-010000.txt", "r+", encoding='utf-8')
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
        infosub = message.split("\t")
        fileidx = infosub[0]
        message = infosub[1]
        message = message.replace("#1", "")
        message = message.replace("#2", "")
        message = message.replace("#3", "")
        message = message.replace("#4", "")
        log(f"{fileidx}\t{message}")
        log(f"\t{pinyins}")

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
                    if (phone_index >= len_pys):
                        print(len_pys)
                        print(phone_index)
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                    else:
                        raise IndexError(f'Unkown PinYin: {pinyin}')
                else:
                    count_phone.append(1)
                    phone_items.append('sp')
            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
            log(f"\t{phone_items_str}")
        except IndexError as e:
            print(f"{fileidx}\t{message}")
            print('except:', e)
            continue

        text = f'[PAD]{message}[PAD]'
        char_embeds = prosody.get_char_embeds(text)
        char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        char_embeds_path = f"./data/berts/{fileidx}.npy"
        np.save(char_embeds_path, char_embeds, allow_pickle=False)

        wave_path = f"./data/waves/{fileidx}.wav"
        spec_path = f"./data/temps/{fileidx}.spec.pt"
        spec = get_spec(hps, wave_path)

        torch.save(spec, spec_path)
        scrips.append(
            f"./data/waves/{fileidx}.wav|./data/temps/{fileidx}.spec.pt|./data/berts/{fileidx}.npy|{phone_items_str}")

    fo.close()

    fout = open(f'./filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[:100]:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/train.txt', 'w', encoding='utf-8')
    for item in scrips[100:]:
        print(item, file=fout)
    fout.close()
