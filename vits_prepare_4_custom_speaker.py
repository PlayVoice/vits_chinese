import os

import pypinyin
import torch
import numpy as np
import argparse
import utils

from bert import TTSProsody
from bert.prosody_tool import is_chinese, pinyin_dict
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
import pypinyin

from vits_pinyin import VITS_PinYin

os.makedirs("./data/waves", exist_ok=True)
os.makedirs("./data/berts", exist_ok=True)
os.makedirs("./data/temps", exist_ok=True)


def log(info: str):
    with open(f'./data/prepare.log', "a", encoding='utf-8') as flog:
        print(info, file=flog)


def get_spec(hps, filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError(
            "{} {} SR doesn't match target {} SR".format(
                sampling_rate, hps.data.sampling_rate
            )
        )
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
    prosody = TTSProsody("./bert", device)
    scrips = []
    speaker = "yueyunyao"

    device = torch.device("cpu")

    # pinyin
    pinyin_generator = VITS_PinYin("./bert", device)

    with open(hps.data.origin_training_files, "r", encoding="utf-8") as f:
        all = f.readlines()
        for i in range(len(all)):
            # 原始数据格式兼容隔壁moegoe，下同:
            # 音频相对路径|说话人ID（单人就是0)|中文文本
            temp = all[i].split("|0|")
            path, content = temp[0], temp[1]
            name = path.split("/")[-1][:-4]
            phone_items_str, char_embeds = pinyin_generator.chinese_to_phonemes(content)
            char_embeds_path = f"./data/berts/{name}.npy"
            np.save(char_embeds_path, char_embeds, allow_pickle=False)

            wave_path = f"./data/waves/{name}.wav"
            spec_path = f"./data/temps/{name}.spec.pt"
            spec = get_spec(hps, wave_path)

            torch.save(spec, spec_path)
            scrips.append(
                f"./data/waves/{name}.wav|./data/temps/{name}.spec.pt|./data/berts/{name}.npy|{phone_items_str}")
            f.close()

    cnt = len(scrips)

    with open(hps.data.origin_validation_files, "r", encoding="utf-8") as f:
        all = f.readlines()
        for i in range(len(all)):
            temp = all[i].split("|0|")
            path, content = temp[0], temp[1]
            name = path.split("/")[-1][:-4]
            phone_items_str, char_embeds = pinyin_generator.chinese_to_phonemes(content)
            char_embeds_path = f"./data/berts/{name}.npy"
            np.save(char_embeds_path, char_embeds, allow_pickle=False)

            wave_path = f"./data/waves/{name}.wav"
            spec_path = f"./data/temps/{name}.spec.pt"
            spec = get_spec(hps, wave_path)

            torch.save(spec, spec_path)
            scrips.append(
                f"./data/waves/{name}.wav|./data/temps/{name}.spec.pt|./data/berts/{name}.npy|{phone_items_str}")
            f.close()

    fout = open(f'./filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[cnt:]:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/train.txt', 'w', encoding='utf-8')
    for item in scrips[:cnt]:
        print(item, file=fout)
    fout.close()
