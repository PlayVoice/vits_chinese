import re

from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from text import pinyin_dict
from bert import TTSProsody


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class VITS_PinYin:
    def __init__(self, bert_path, device):
        self.pinyin_parser = Pinyin(MyConverter())
        self.prosody = TTSProsody(bert_path, device)

    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    def chinese_to_phonemes(self, text):
        phonemes = ["sil"]
        text = text.replace("、", ",")
        text = text.replace("，", ",")
        text = text.replace(":", ",")
        text = text.replace(";", ",")
        text = text.replace("!", ",")
        text = text.replace("?", ",")
        chars = ['[PAD]']
        count_phone = []
        count_phone.append(1)
        for subtext in text.split(","):
            if (len(subtext) == 0):
                continue
            pinyins = self.correct_pinyin_tone3(subtext)
            sub_p, sub_c = self.get_phoneme4pinyin(pinyins)
            phonemes.extend(sub_p)
            phonemes.append("sp")
            count_phone.extend(sub_c)
            count_phone.append(1)
            chars.append(subtext)
            chars.append(',')
        phonemes.append("sil")
        count_phone.append(1)
        chars.append('[PAD]')
        chars = "".join(chars)
        char_embeds = self.prosody.get_char_embeds(chars)
        char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return " ".join(phonemes), char_embeds

    def correct_pinyin_tone3(self, text):
        pinyin_list = [p[0] for p in self.pinyin_parser.pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
        if len(pinyin_list) >= 2:
            for i in range(1, len(pinyin_list)):
                try:
                    if re.findall(r'\d', pinyin_list[i-1])[0] == '3' and re.findall(r'\d', pinyin_list[i])[0] == '3':
                        pinyin_list[i-1] = pinyin_list[i-1].replace('3', '2')
                except IndexError:
                    pass
        return pinyin_list
