### Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

https://user-images.githubusercontent.com/16432329/220678182-4775dec8-9229-4578-870f-2eebc3a5d660.mp4


Based on BERT, NaturalSpeech, VITS

### Features
1, Hidden prosody embedding from BERT，get natural pauses in grammar

2, Infer loss from NaturalSpeech，get less sound error

3, Framework of VITS，get high audio quality

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

### Train
download baker data: https://www.data-baker.com/data/index/TNtts/

change sample rate of waves to **16kHz**, and put waves to ./data/waves

put 000001-010000.txt to ./data/000001-010000.txt

> python vits_prepare.py -c ./configs/bert_vits.json

> python train.py -c configs/bert_vits.json -m bert_vits


![bert_lose](https://user-images.githubusercontent.com/16432329/220883346-c382bea2-1d2f-4a16-b797-2f9e2d2fb639.png)

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

https://github.com/Executedone/Chinese-FastSpeech2

https://github.com/jaywalnut310/vits

### Voice Clone

![vits_clone](https://user-images.githubusercontent.com/16432329/223632792-519178d4-382d-4df4-93d7-088b870cfc42.jpg)

TODO ~


### Reference For Voice Clone
[Speak, Read and Prompt:High-Fidelity Text-to-Speech with Minimal Supervision](https://arxiv.org/abs/2302.03540)

[HierSpeech: Bridging the Gap between Text andSpeech by Hierarchical Variational Inference usingSelf-supervised Representations for Speech Synthesis](https://openreview.net/forum?id=awdyRVnfQKX)

[Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus](https://github.com/hcy71o/TransferTTS)

[AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation](https://arxiv.org/abs/2206.00208)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[Residual Adapters for Few-Shot Text-to-Speech Speaker Adaptation](https://arxiv.org/abs/2210.15868)

https://github.com/collabora/spear-tts-pytorch

https://github.com/CODEJIN/HierSpeech
