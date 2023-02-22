import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class CharEmbedding(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.bert = BertModel(self.bert_config)
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)

    def text2Token(self, text):
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)
        return out_seq


class TTSProsody(object):
    def __init__(self, path, device):
        self.device = device
        self.char_model = CharEmbedding(path)
        self.char_model.load_state_dict(
            torch.load(
                os.path.join(path, 'prosody_model.pt'),
                map_location="cpu"
            ),
            strict=False
        )
        self.char_model.eval()
        self.char_model.to(self.device)

    def get_char_embeds(self, text):
        input_ids = self.char_model.text2Token(text)
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)
        input_ids = torch.LongTensor([input_ids]).to(self.device)
        input_masks = torch.LongTensor([input_masks]).to(self.device)
        type_ids = torch.LongTensor([type_ids]).to(self.device)

        with torch.no_grad():
            char_embeds = self.char_model(
                input_ids, input_masks, type_ids).squeeze(0).cpu()
        return char_embeds

    def expand_for_phone(self, char_embeds, length):  # length of phones for char
        assert char_embeds.size(0) == len(length)
        expand_vecs = list()
        for vec, leng in zip(char_embeds, length):
            vec = vec.expand(leng, -1)
            expand_vecs.append(vec)
        expand_embeds = torch.cat(expand_vecs, 0)
        assert expand_embeds.size(0) == sum(length)
        return expand_embeds.numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prosody = TTSProsody('./bert/', device)
    while True:
        text = input("请输入文本：")
        prosody.get_char_embeds(text)
