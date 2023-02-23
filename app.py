from models import SynthesizerTrn
from vits_pinyin import VITS_PinYin
from text import cleaned_text_to_sequence
from text.symbols import symbols
import gradio as gr
import utils
import torch
import argparse
import os
import re
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"


def create_calback(net_g: SynthesizerTrn, tts_front: VITS_PinYin):
    def tts_calback(text, dur_scale):
        if limitation:
            text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
            max_len = 150
            if text_len > max_len:
                return "Error: Text is too long", None

        phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(
                char_embeds).unsqueeze(0).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
                                length_scale=dur_scale)[0][0, 0].data.cpu().float().numpy()
        del x_tst, x_tst_lengths, x_tst_prosody
        return "Success", (16000, audio)

    return tts_calback


example = [['天空呈现的透心的蓝，像极了当年。总在这样的时候，透过窗棂，心，在天空里无尽的游弋！柔柔的，浓浓的，痴痴的风，牵引起心底灵动的思潮；情愫悠悠，思情绵绵，风里默坐，红尘中的浅醉，诗词中的优柔，任那自在飞花轻似梦的情怀，裁一束霓衣，织就清浅淡薄的安寂。', 1],
           ['风的影子翻阅过淡蓝色的信笺，柔和的文字浅浅地漫过我安静的眸，一如几朵悠闲的云儿，忽而氤氲成汽，忽而修饰成花，铅华洗尽后的透彻和靓丽，爽爽朗朗，轻轻盈盈', 1],
           ['时光仿佛有穿越到了从前，在你诗情画意的眼波中，在你舒适浪漫的暇思里，我如风中的思绪徜徉广阔天际，仿佛一片沾染了快乐的羽毛，在云环影绕颤动里浸润着风的呼吸，风的诗韵，那清新的耳语，那婉约的甜蜜，那恬淡的温馨，将一腔情澜染得愈发的缠绵。', 1],]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")
    args = parser.parse_args()

    device = torch.device("cpu")

    # pinyin
    tts_front = VITS_PinYin("./bert", device)

    # config
    hps = utils.get_hparams_from_file("./configs/bert_vits.json")

    # model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)

    model_path = "vits_bert_model.pth"
    utils.load_model(model_path, net_g)
    net_g.eval()
    net_g.to(device)

    tts_calback = create_calback(net_g, tts_front)

    app = gr.Blocks()
    with app:
        gr.Markdown("# Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft\n\n"
                    "code : github.com/PlayVoice/vits_chinese\n\n"
                    "1, Hidden prosody embedding from BERT，get natural pauses in grammar\n\n"
                    "2, Infer loss from NaturalSpeech，get less sound error\n\n"
                    "3, Framework of VITS，get high audio quality\n\n"
                    "<video id='video' controls='' preload='yes'>\n\n"
                    "<source id='mp4' src='https://user-images.githubusercontent.com/16432329/220678182-4775dec8-9229-4578-870f-2eebc3a5d660.mp4' type='video/mp4'>\n\n"
                    "</videos>\n\n"
                    )

        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        textbox = gr.TextArea(label="Text",
                                              placeholder="Type your sentence here (Maximum 150 words)",
                                              value="中文语音合成", elem_id=f"tts-input")
                        duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                    label='速度 Speed')
                    with gr.Column():
                        text_output = gr.Textbox(label="Message")
                        audio_output = gr.Audio(
                            label="Output Audio", elem_id="tts-audio")
                        btn = gr.Button("Generate!")
                        btn.click(tts_calback,
                                  inputs=[textbox, duration_slider],
                                  outputs=[text_output, audio_output])
            gr.Examples(
                examples=example,
                inputs=[textbox, duration_slider],
                outputs=[text_output, audio_output],
                fn=tts_calback
            )
    app.queue(concurrency_count=3).launch(show_api=False, share=args.share)
