#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import soundfile
import torch
import os
import torch
import argparse

from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin


def display(sess):
    for i in sess.get_inputs():
        print(i)

    print("-" * 10)
    for o in sess.get_outputs():
        print(o)


class OnnxModel:
    def __init__(
        self,
        model: str,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.model = onnxruntime.InferenceSession(
            model,
            sess_options=self.session_opts,
        )
        display(self.model)

        meta = self.model.get_modelmeta().custom_metadata_map
        self.add_blank = int(meta["add_blank"])
        self.sample_rate = int(meta["sample_rate"])
        print(meta)

    def __call__(self, x: torch.Tensor):
        """
        Args:
          x:
            A int64 tensor of shape (L,)
        """
        x = x.unsqueeze(0)
        x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
        noise_scale = torch.tensor([1], dtype=torch.float32)
        length_scale = torch.tensor([1], dtype=torch.float32)

        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_length.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
            },
        )[0]
        return y


def main():
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    print("Onnx model path:", args.model)
    model = OnnxModel(args.model)

    tts_front = VITS_PinYin(None, None, hasBert=False)

    os.makedirs("./vits_infer_out/", exist_ok=True)

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
        phonemes, _ = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)

        x = torch.tensor(input_ids, dtype=torch.int64)
        y = model(x)

        soundfile.write(
            f"./vits_infer_out/onnx_{n}.wav", y, model.sample_rate)

    fo.close()


if __name__ == "__main__":
    main()
