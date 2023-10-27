#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import soundfile
import torch
import os
import torch
import argparse
import datetime
import numpy

from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin


def display(sess):
    for i in sess.get_inputs():
        print(i)

    print("-" * 10)
    for o in sess.get_outputs():
        print(o)


class OnnxModel_Encoder:
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

        z_p, y_mask = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_length.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
            },
        )
        return z_p, y_mask


class OnnxModel_Decoder:
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
        self.hop_length = int(meta["hop_length"])
        print(meta)

    def __call__(self, z_p, y_mask):
        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: z_p,
                self.model.get_inputs()[1].name: y_mask,
            },
        )[0]
        return y


def main_debug():
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--decoder', type=str, required=True)
    args = parser.parse_args()
    print("Onnx model path:", args.encoder)
    print("Onnx model path:", args.decoder)

    encoder = OnnxModel_Encoder(args.encoder)
    decoder = OnnxModel_Decoder(args.decoder)

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
        print(n)
        print(datetime.datetime.now())
        phonemes, _ = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)

        x = torch.tensor(input_ids, dtype=torch.int64)
        z_p, y_mask = encoder(x)
        y = decoder(z_p, y_mask)
        print(datetime.datetime.now())

        soundfile.write(
            f"./vits_infer_out/onnx_stream_{n}.wav", y, encoder.sample_rate)

    fo.close()


def main():
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--decoder', type=str, required=True)
    args = parser.parse_args()
    print("Onnx model path:", args.encoder)
    print("Onnx model path:", args.decoder)

    encoder = OnnxModel_Encoder(args.encoder)
    decoder = OnnxModel_Decoder(args.decoder)

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
        print(n)
        print(datetime.datetime.now())
        phonemes, _ = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)

        x = torch.tensor(input_ids, dtype=torch.int64)
        z_p, y_mask = encoder(x)
        print(datetime.datetime.now())
        len_z = z_p.shape[2]
        print('frame size is: ', len_z)
        print('hop_length is: ', decoder.hop_length)
        # can not change these parameters
        hop_length = decoder.hop_length
        hop_frame = 12
        hop_sample = hop_frame * hop_length
        stream_chunk = 50
        stream_index = 0
        stream_out_wav = []

        while (stream_index < len_z):
            if (stream_index == 0):  # start frame
                cut_s = stream_index
                cut_s_wav = 0
            else:
                cut_s = stream_index - hop_frame
                cut_s_wav = hop_sample

            if (stream_index + stream_chunk > len_z - hop_frame):  # end frame
                cut_e = stream_index + stream_chunk
                cut_e_wav = -1
            else:
                cut_e = stream_index + stream_chunk + hop_frame
                cut_e_wav = -1 * hop_sample

            z_chunk = z_p[:, :, cut_s:cut_e]
            m_chunk = y_mask[:, :, cut_s:cut_e]
            o_chunk = decoder(z_chunk, m_chunk)
            o_chunk = o_chunk[cut_s_wav:cut_e_wav]
            stream_out_wav.extend(o_chunk)
            stream_index = stream_index + stream_chunk
            print(datetime.datetime.now())

        stream_out_wav = numpy.asarray(stream_out_wav)
        soundfile.write(
            f"./vits_infer_out/onnx_stream_{n}.wav", stream_out_wav, encoder.sample_rate)

    fo.close()


if __name__ == "__main__":
    main()
