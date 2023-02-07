import os
import sys
import numpy as np

from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile

import datetime

import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, pinyin_dict


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def load_pinyin_dict():
    my_dict = {}
    with open("./misc/pypinyin-local.dict", "r", encoding="utf-8") as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            pinyin = cuts[1:]
            tmp = []
            for one in pinyin:
                onelist = [one]
                tmp.append(onelist)
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


def get_phoneme4pinyin(pinyins):
    result = []
    for pinyin in pinyins:
        if pinyin[:-1] in pinyin_dict:
            tone = pinyin[-1]
            a = pinyin[:-1]
            a1, a2 = pinyin_dict[a]
            result += [a1, a2 + tone, "#0"]
    result.append("sil")
    return result


def chinese_to_phonemes(pinyin_parser, text):
    phonemes = ["sil"]
    for subtext in text.split("，"):
        pinyins = pinyin_parser.pinyin(subtext, style=Style.TONE3, errors="ignore")
        new_pinyin = []
        for x in pinyins:
            x = "".join(x)
            new_pinyin.append(x)
        sub_phonemes = get_phoneme4pinyin(new_pinyin)
        phonemes.extend(sub_phonemes)
    phonemes.append("eos")
    # print(f"phoneme seq: {phonemes}")
    return " ".join(phonemes)


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def get_text(phones, hps):
    text_norm = cleaned_text_to_sequence(phones)
    # baker 应该将add_blank设置为false
    # [0, 19, 81, 3, 14, 51, 3, 0, 1]
    # [0, 0, 0, 19, 0, 81, 0, 3, 0, 14, 0, 51, 0, 3, 0, 0, 0, 1, 0]
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


#
load_pinyin_dict()
#
pinyin_parser = Pinyin(MyConverter())
#

# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/baker_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).cuda()

_ = utils.load_checkpoint("./logs/baker_base/G_200000.pth", net_g, None)
net_g.eval()
net_g.remove_weight_norm()

# check directory existence
if not os.path.exists("./vits_out"):
    os.makedirs("./vits_out")

if __name__ == "__main__":
    n = 0
    fo = open("vits_strings.txt", "r+")
    while True:
        try:
            message = fo.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        n = n + 1

        print("===============================================================")
        phonemes = chinese_to_phonemes(pinyin_parser, message)
        # phonemes = phonemes.replace("^ ", "")
        #
        input_ids = get_text(phonemes, hps)

        print(datetime.datetime.now())
        with torch.no_grad():
            x_tst = input_ids.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([input_ids.size(0)]).cuda()
            audio = (
                net_g.infer(
                    x_tst, x_tst_lengths, noise_scale=0, length_scale=1
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        print(datetime.datetime.now())

        save_wav(audio, f"./vits_out/{n}_baker.wav", hps.data.sampling_rate)

        print(message)
        print(phonemes)
        print(input_ids)
    fo.close()

    # can be deleted
    os.system("chmod 777 ./vits_out -R")
