# Best practice TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/vits_chinese)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/vits_chinese">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/vits_chinese">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/vits_chinese">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/vits_chinese">

# 这是一个用于TTS算法学习的项目，如果您在寻找直接用于生产的TTS，本项目可能不适合您！
https://user-images.githubusercontent.com/16432329/220678182-4775dec8-9229-4578-870f-2eebc3a5d660.mp4


Based on BERT, NaturalSpeech, VITS

### Features
1, Hidden prosody embedding from BERT，get natural pauses in grammar

2, Infer loss from NaturalSpeech，get less sound error

3, Framework of VITS，get high audio quality

:heartpulse::heartpulse::heartpulse:Tip: It is recommended to use **Infer Loss** fine-tune model after base model trained, and freeze **PosteriorEncoder** during fine-tuning.

### Online demo
https://huggingface.co/spaces/maxmax20160403/vits_chinese

### Install

> pip install -r requirements.txt

> cd monotonic_align

> python setup.py build_ext --inplace

### Infer with Pretrained model

BaiduYun: https://pan.baidu.com/s/1Cj4MnwFyZ0XZmTR6EpygbQ?pwd=yn60

Google: https://drive.google.com/drive/folders/1sioiNpebOLyCmHURgOgJ7ppWI7b-7Rb5?usp=sharing

Or get from release page

put prosody_model.pt To ./bert/prosody_model.pt

put vits_bert_model.pth To ./vits_bert_model.pth

> python vits_infer.py --config ./configs/bert_vits.json --model vits_bert_model.pth

./vits_infer_out have the waves infered, listen !!!

### Infer with chunk wave streaming out

as key paramter, ***hop_frame = ∑decoder.ups.padding***

> python vits_infer_stream.py --config ./configs/bert_vits.json --model vits_bert_model.pth

### Text normlize

> pip install WeTextProcessing

> from tn.chinese.normalizer import Normalizer

> normalizer = Normalizer()

> ...

> item = fo.readline().strip()

> item = normalizer.normalize(item)

### Train
download baker data: https://www.data-baker.com/data/index/TNtts/

change sample rate of waves to **16kHz**, and put waves to ./data/waves

put 000001-010000.txt to ./data/000001-010000.txt

> python vits_prepare.py -c ./configs/bert_vits.json

> python train.py -c configs/bert_vits.json -m bert_vits


![bert_lose](https://user-images.githubusercontent.com/16432329/220883346-c382bea2-1d2f-4a16-b797-2f9e2d2fb639.png)

### 额外说明

原始标注为
``` c
000001	卡尔普#2陪外孙#1玩滑梯#4。
  ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002	假语村言#2别再#1拥抱我#4。
  jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
```

需要标注为，BERT需要汉字 `卡尔普陪外孙玩滑梯。` (包括标点)，TTS需要声韵母 `sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil`
``` c
000001	卡尔普陪外孙玩滑梯。
  ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
  sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil
000002	假语村言别再拥抱我。
  jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
  sil j ia2 ^ v3 c uen1 ^ ian2 b ie2 z ai4 ^ iong1 b ao4 ^ uo3 sp sil
```

训练标注为
```
./data/wavs/000001.wav|./data/mels/000001.pt|./data/berts/000001.npy|sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil
./data/wavs/000002.wav|./data/mels/000002.pt|./data/berts/000002.npy|sil j ia2 ^ v3 c uen1 ^ ian2 b ie2 z ai4 ^ iong1 b ao4 ^ uo3 sp sil
```

遇到这句话会出错
```
002365	这图#2难不成#2是#1Ｐ过的#4？
  zhe4 tu2 nan2 bu4 cheng2 shi4 P IY1 guo4 de5
```

### 拼音错误修改
将正确的词语和拼音写入文件： [./text/pinyin-local.txt](./text/pinyin-local.txt)
```
渐渐 jian4 jian4
浅浅 qian3 qian3
```

### Model compression based on knowledge distillation
Student model has 53M size and 3× speed of teacher model.

To train:

> python train.py -c configs/bert_vits_student.json -m bert_vits_student

To infer, get studet model at release page or 

Google: :https://drive.google.com/file/d/1hTLWYEKH4GV9mQltrMyr3k2UKUo4chdp/view?usp=sharing

> python vits_infer.py --config ./configs/bert_vits_student.json --model vits_bert_student.pth

You can use vits_istft as a student model too.

https://github.com/PlayVoice/vits_chinese/tree/vits_istft

### Video text
> 天空呈现的透心的蓝，像极了当年。总在这样的时候，透过窗棂，心，在天空里无尽的游弋！柔柔的，浓浓的，痴痴的风，牵引起心底灵动的思潮；情愫悠悠，思情绵绵，风里默坐，红尘中的浅醉，诗词中的优柔，任那自在飞花轻似梦的情怀，裁一束霓衣，织就清浅淡薄的安寂。
> 
> 风的影子翻阅过淡蓝色的信笺，柔和的文字浅浅地漫过我安静的眸，一如几朵悠闲的云儿，忽而氤氲成汽，忽而修饰成花，铅华洗尽后的透彻和靓丽，爽爽朗朗，轻轻盈盈
> 
> 时光仿佛有穿越到了从前，在你诗情画意的眼波中，在你舒适浪漫的暇思里，我如风中的思绪徜徉广阔天际，仿佛一片沾染了快乐的羽毛，在云环影绕颤动里浸润着风的呼吸，风的诗韵，那清新的耳语，那婉约的甜蜜，那恬淡的温馨，将一腔情澜染得愈发的缠绵。

### Reference For TTS
[Microsoft's NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421)

https://github.com/Executedone/Chinese-FastSpeech2 **bert prosody**

https://github.com/wenet-e2e/WeTextProcessing

https://github.com/jaywalnut310/vits

### Info For Voice Clone
[Speak, Read and Prompt:High-Fidelity Text-to-Speech with Minimal Supervision](https://arxiv.org/abs/2302.03540)

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://arxiv.org/pdf/2211.16866.pdf)

[HierSpeech: Bridging the Gap between Text andSpeech by Hierarchical Variational Inference usingSelf-supervised Representations for Speech Synthesis](https://openreview.net/forum?id=awdyRVnfQKX)

[Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus](https://github.com/hcy71o/TransferTTS)

[AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation](https://arxiv.org/abs/2206.00208)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[Residual Adapters for Few-Shot Text-to-Speech Speaker Adaptation](https://arxiv.org/abs/2210.15868)

https://github.com/collabora/spear-tts-pytorch

https://github.com/CODEJIN/HierSpeech

https://github.com/hcy71o/SNAC
