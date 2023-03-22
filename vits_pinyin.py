import re

import pypinyin
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

import numpy as np

from text import pinyin_dict
from bert import TTSProsody


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if (is_chinese(char)):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean


class VITS_PinYin:
    def __init__(self, bert_path, device):
        self.pinyin_parser = Pinyin(MyConverter())
        self.prosody = TTSProsody(bert_path, device)

    # def get_phoneme4pinyin(self, pinyins):
    #     result = []
    #     count_phone = []
    #     for pinyin in pinyins:
    #         if pinyin[:-1] in pinyin_dict:
    #             tone = pinyin[-1]
    #             a = pinyin[:-1]
    #             a1, a2 = pinyin_dict[a]
    #             result += [a1, a2 + tone]
    #             count_phone.append(2)
    #     return result, count_phone

    def chinese_to_phonemes(self, text):
        # @todo:考虑使用g2pw的chinese bert替换原始的pypinyin,目前测试下来运行速度太慢。
        # 将标准中文文本符号替换成 bert 符号库中的单符号,以保证bert的效果.
        text = text.replace("——", "...")\
            .replace("—", "...")\
            .replace("……", "...")\
            .replace("…", "...")\
            .replace('“', '"')\
            .replace('”', '"')\
            .replace("\n", "")
        tokens = self.prosody.char_model.tokenizer.tokenize(text)
        assert not tokens.count("[UNK]")
        pinyins = np.reshape(pypinyin.pinyin(text, style=pypinyin.TONE3), (-1))
        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)
            temp = ""

            len_pys = len(tokens)
            for word in tokens:
                if is_chinese(word):
                    count_phone.append(2)
                    if (phone_index >= len_pys):
                        print(f"!!!![{text}]plz check ur text whether includes MULTIBYTE symbol.(请检查你的文本中是否包含多字节符号)")
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if not pinyin[-1].isdigit():
                        pinyin += "5"
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                else:
                    temp += word
                    if temp == pinyins[phone_index]:
                        temp = ""
                        phone_index += 1
                    count_phone.append(1)
                    phone_items.append('sp')

            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
        except IndexError as e:
            print('except:', e)

        text = f'[PAD]{text}[PAD]'
        char_embeds = self.prosody.get_char_embeds(text)
        char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return phone_items_str, char_embeds

    # def correct_pinyin_tone3(self, text):
    #     pinyin_list = [p[0] for p in self.pinyin_parser.pinyin(
    #         text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
    #     if len(pinyin_list) >= 2:
    #         for i in range(1, len(pinyin_list)):
    #             try:
    #                 if re.findall(r'\d', pinyin_list[i - 1])[0] == '3' and re.findall(r'\d', pinyin_list[i])[0] == '3':
    #                     pinyin_list[i - 1] = pinyin_list[i - 1].replace('3', '2')
    #             except IndexError:
    #                 pass
    #     return pinyin_list
