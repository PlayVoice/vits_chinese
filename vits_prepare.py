import os
import torch
import numpy as np

from bert import TTSProsody
from bert.prosody_tool import is_chinese, pinyin_dict


os.makedirs("./data/waves", exist_ok=True)
os.makedirs("./data/berts", exist_ok=True)


def log(info: str):
    with open(f'./data/prepare.log', "a", encoding='utf-8') as flog:
        print(info, file=flog)


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    prosody = TTSProsody("./bert", device)

    fo = open(f"./data/000001-010000.txt", "r+")
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

        #text = f'[PAD]{message}[PAD]'
        #char_embeds = prosody.get_char_embeds(text)
        #char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        #char_embeds_path = f"./data/berts/{fileidx}.npy"
        #np.save(char_embeds_path, char_embeds, allow_pickle=False)
        scrips.append(f"./data/waves/{fileidx}.wav|./data/berts/{fileidx}.npy|{phone_items_str}")

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
