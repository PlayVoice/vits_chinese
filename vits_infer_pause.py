import os
import sys
import numpy as np

import torch
import utils
import argparse

from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin

parser = argparse.ArgumentParser(description='Inference code for bert vits models')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--pause', type=int, required=True)
args = parser.parse_args()

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

# device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# pinyin
tts_front = VITS_PinYin("./bert", device)

# config
hps = utils.get_hparams_from_file(args.config)

# model
net_g = utils.load_class(hps.train.eval_class)(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)

# model_path = "logs/bert_vits/G_200000.pth"
# utils.save_model(net_g, "vits_bert_model.pth")
# model_path = "vits_bert_model.pth"
utils.load_model(args.model, net_g)
net_g.eval()
net_g.to(device)

os.makedirs("./vits_infer_out/", exist_ok=True)
if __name__ == "__main__":
    n = 0
    fo = open("vits_infer_item.txt", "r+", encoding='utf-8')
    while (True):
        try:
            item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (item == None or item == ""):
            break
        n = n + 1
        phonemes, char_embeds = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        pause_tmpt = np.array(input_ids)
        pause_mask = np.where(pause_tmpt == 2, 0, 1)
        pause_valu = np.where(pause_tmpt == 2, 1, 0)
        assert args.pause > 1
        pause_valu = pause_valu * ((args.pause * 16) // 256)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
            audio = net_g.infer_pause(x_tst, x_tst_lengths, x_tst_prosody, pause_mask, pause_valu, noise_scale=0.5,
                                length_scale=1)[0][0, 0].data.cpu().float().numpy()
        save_wav(audio, f"./vits_infer_out/bert_vits_{n}.wav", hps.data.sampling_rate)
    fo.close()
