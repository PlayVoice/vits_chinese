# Best practice TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/vits_chinese)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/vits_chinese">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/vits_chinese">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/vits_chinese">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/vits_chinese">

## 这是一个用于TTS算法学习的项目，如果您在寻找直接用于生产的TTS，本项目可能不适合您！
https://user-images.githubusercontent.com/16432329/220678182-4775dec8-9229-4578-870f-2eebc3a5d660.mp4

> 天空呈现的透心的蓝，像极了当年。总在这样的时候，透过窗棂，心，在天空里无尽的游弋！柔柔的，浓浓的，痴痴的风，牵引起心底灵动的思潮；情愫悠悠，思情绵绵，风里默坐，红尘中的浅醉，诗词中的优柔，任那自在飞花轻似梦的情怀，裁一束霓衣，织就清浅淡薄的安寂。
> 
> 风的影子翻阅过淡蓝色的信笺，柔和的文字浅浅地漫过我安静的眸，一如几朵悠闲的云儿，忽而氤氲成汽，忽而修饰成花，铅华洗尽后的透彻和靓丽，爽爽朗朗，轻轻盈盈
> 
> 时光仿佛有穿越到了从前，在你诗情画意的眼波中，在你舒适浪漫的暇思里，我如风中的思绪徜徉广阔天际，仿佛一片沾染了快乐的羽毛，在云环影绕颤动里浸润着风的呼吸，风的诗韵，那清新的耳语，那婉约的甜蜜，那恬淡的温馨，将一腔情澜染得愈发的缠绵。

### Features，特性
1, Hidden prosody embedding from **BERT**，get natural pauses in grammar

2, Infer loss from **NaturalSpeech**，get less sound error

3, Framework of **VITS**，get high audio quality

4, Module-wise Distillation, get speedup

:heartpulse:**Tip**: It is recommended to use **Infer Loss** fine-tune model after base model trained, and freeze **PosteriorEncoder** during fine-tuning.

:heartpulse:**意思就是：初步训练时，不用loss_kl_r；训练好后，添加loss_kl_r继续训练，稍微训练一下就行了，如果音频质量差，可以给loss_kl_r乘以一个小于1的系数、降低loss_kl_r对模型的贡献；继续训练时，可以尝试冻结音频编码器Posterior Encoder；总之，玩法很多，需要多尝试！**

<div align="center">
	
![naturalspeech](https://github.com/PlayVoice/vits_chinese/assets/16432329/0d7ceb00-f159-40a4-8897-b3f2a3c824d3)

</div>

### 为什么不升级为VITS2
VITS2最重要的改进是将Flow的WaveNet模块使用Transformer替换，而在TTS流式实现中，通常需要用纯CNN替换Transformer。

### Online demo，在线体验
https://huggingface.co/spaces/maxmax20160403/vits_chinese

### Install，安装依赖和MAS对齐

> pip install -r requirements.txt

> cd monotonic_align

> python setup.py build_ext --inplace

### Infer with Pretrained model，用示例模型推理

Get from release page [vits_chinese/releases/](https://github.com/PlayVoice/vits_chinese/releases/tag/v1.0)

put [prosody_model.pt](https://github.com/PlayVoice/vits_chinese/releases/tag/v1.0) To ./bert/prosody_model.pt

put [vits_bert_model.pth](https://github.com/PlayVoice/vits_chinese/releases/tag/v1.0) To ./vits_bert_model.pth

```
python vits_infer.py --config ./configs/bert_vits.json --model vits_bert_model.pth
```

./vits_infer_out have the waves inferred, listen !!!

### Infer with chunk wave streaming out，分块流式推理

as key parameter, ***hop_frame = ∑decoder.ups.padding*** :heartpulse:

```
python vits_infer_stream.py --config ./configs/bert_vits.json --model vits_bert_model.pth
```

### Ceil duration affect naturalness
So change **w_ceil = torch.ceil(w)** to **w_ceil = torch.ceil(w + 0.35)**

### All Thanks To Our Contributors:
<a href="https://github.com/MaxMax2016/vits_chinese/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MaxMax2016/vits_chinese" />
</a>

### Train，训练
download baker data [https://aistudio.baidu.com/datasetdetail/36741](https://aistudio.baidu.com/datasetdetail/36741), more info: https://www.data-baker.com/data/index/TNtts/

change sample rate of waves to **16kHz**, and put waves to ./data/waves

```
python vits_resample.py -w [input path]:[./data/Wave/] -o ./data/waves -s 16000
```

put 000001-010000.txt to ./data/000001-010000.txt

```
python vits_prepare.py -c ./configs/bert_vits.json
```

```
python train.py -c configs/bert_vits.json -m bert_vits
```

![bert_lose](https://user-images.githubusercontent.com/16432329/220883346-c382bea2-1d2f-4a16-b797-2f9e2d2fb639.png)

### 额外说明

原始标注为
``` c
000001	卡尔普#2陪外孙#1玩滑梯#4。
  ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002	假语村言#2别再#1拥抱我#4。
  jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
```

标注规整后：
- BERT需要汉字 `卡尔普陪外孙玩滑梯。` (包括标点)
- TTS需要声韵母 `sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil`
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
./data/wavs/000001.wav|./data/temps/000001.spec.pt|./data/berts/000001.npy|sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil
./data/wavs/000002.wav|./data/temps/000002.spec.pt|./data/berts/000002.npy|sil j ia2 ^ v3 c uen1 ^ ian2 b ie2 z ai4 ^ iong1 b ao4 ^ uo3 sp sil
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

### 数字播报支持
已支持，基于WeNet开源社区[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)；当然，不可能是完美的

### 不使用Bert也能推理
```
python vits_infer_no_bert.py --config ./configs/bert_vits.json --model vits_bert_model.pth
```
虽然训练使用了Bert，但推理可以完全不用Bert，牺牲自然停顿来适配低计算资源设备，比如手机

低资源设备通常会分句合成，这样牺牲的自然停顿也没那么明显

### ONNX非流式
导出：会有许多警告，直接忽略
```
python model_onnx.py --config configs/bert_vits.json --model vits_bert_model.pth
```
推理
```
python vits_infer_onnx.py --model vits-chinese.onnx
```

### ONNX流式

具体实现，将VITS拆解为两个模型，取名为Encoder和Decoder。

- Encoder包括TextEncoder与DurationPredictor等；

- Decoder包括ResidualCouplingBlock与Generator等；

- ResidualCouplingBlock，即Flow，可以放在Encoder或Decoder，放在Decoder需要更大的**hop_frame**

并且将推理逻辑也进行了切分；特别的，先验分布的采样过程放在了Encoder中：
```
z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
```

ONNX流式模型导出
```
python model_onnx_stream.py --config configs/bert_vits.json --model vits_bert_model.pth
```

ONNX流式模型推理
```
python vits_infer_onnx_stream.py --encoder vits-chinese-encoder.onnx --decoder vits-chinese-decoder.onnx
```

在流式推理中，**hop_frame**是一个重要参数，需要去尝试出合适的值

### Model compression based on knowledge distillation，应该叫迁移学习还是知识蒸馏呢？
Student model has 53M size and 3× speed of teacher model.

To train:

```
python train.py -c configs/bert_vits_student.json -m bert_vits_student
```

To infer, get [student model](https://github.com/PlayVoice/vits_chinese/releases/tag/v2.0) at the release page

```
python vits_infer.py --config ./configs/bert_vits_student.json --model vits_bert_student.pth
```

### 代码来源
[Microsoft's NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421)

[Nix-TTS: Lightweight and End-to-End Text-to-Speech via Module-wise Distillation](https://arxiv.org/abs/2203.15643)

https://github.com/Executedone/Chinese-FastSpeech2 **bert prosody**

https://github.com/wenet-e2e/WeTextProcessing

[https://github.com/TensorSpeech/TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/processor/baker.py) **Heavily depend on**

https://github.com/jaywalnut310/vits

https://github.com/wenet-e2e/wetts

https://github.com/csukuangfj **onnx and android**

### BERT应用于TTS
2019 BERT+Tacotron2: Pre-trained text embeddings for enhanced text-tospeech synthesis.

2020 BERT+Tacotron2-MultiSpeaker: Improving prosody with linguistic and bert derived features in multi-speaker based mandarin chinese neural tts.

2021 BERT+Tacotron2: Extracting and predicting word-level style variations for speech synthesis.

2022 https://github.com/Executedone/Chinese-FastSpeech2

2023 BERT+VISINGER: Towards Improving the Expressiveness of Singing Voice Synthesis with BERT Derived Semantic Information.

# AISHELL3多发音人训练，训练出的模型可用于克隆
切换代码分支[bert_vits_aishell3](https://github.com/PlayVoice/vits_chinese/tree/bert_vits_aishell3)，对比分支代码可以看到**针对多发音人所做出的修改**

## 数据下载
http://www.openslr.org/93/

## 采样率转换
```
python prep_resample.py --wav aishell-3/train/wav/ --out vits_data/waves-16k
```

## 标注规范化（labels.txt，名称不能改）
```
python prep_format_label.py --txt aishell-3/train/content.txt --out vits_data/labels.txt
```

- 原始标注
```
SSB00050001.wav	广 guang3 州 zhou1 女 nv3 大 da4 学 xue2 生 sheng1 登 deng1 山 shan1 失 shi1 联 lian2 四 si4 天 tian1 警 jing3 方 fang1 找 zhao3 到 dao4 疑 yi2 似 si4 女 nv3 尸 shi1
SSB00050002.wav	尊 zhun1 重 zhong4 科 ke1 学 xue2 规 gui1 律 lv4 的 de5 要 yao1 求 qiu2
SSB00050003.wav	七 qi1 路 lu4 无 wu2 人 ren2 售 shou4 票 piao4
```
- 规范标注
```
SSB00050001.wav 广州女大学生登山失联四天警方找到疑似女尸
	guang3 zhou1 nv3 da4 xue2 sheng1 deng1 shan1 shi1 lian2 si4 tian1 jing3 fang1 zhao3 dao4 yi2 si4 nv3 shi1
SSB00050002.wav 尊重科学规律的要求
	zhun1 zhong4 ke1 xue2 gui1 lv4 de5 yao1 qiu2
SSB00050003.wav 七路无人售票
	qi1 lu4 wu2 ren2 shou4 piao4
```
## 数据预处理
```
python prep_bert.py --conf configs/bert_vits.json --data vits_data/
```

打印信息，在过滤本项目不支持的**儿化音**

生成 vits_data/speakers.txt
```
{'SSB0005': 0, 'SSB0009': 1, 'SSB0011': 2..., 'SSB1956': 173}
```
生成 filelists
```
0|vits_data/waves-16k/SSB0005/SSB00050001.wav|vits_data/temps/SSB0005/SSB00050001.spec.pt|vits_data/berts/SSB0005/SSB00050001.npy|sil g uang3 zh ou1 n v3 d a4 x ve2 sh eng1 d eng1 sh an1 sh iii1 l ian2 s ii4 t ian1 j ing3 f ang1 zh ao3 d ao4 ^ i2 s ii4 n v3 sh iii1 sil
0|vits_data/waves-16k/SSB0005/SSB00050002.wav|vits_data/temps/SSB0005/SSB00050002.spec.pt|vits_data/berts/SSB0005/SSB00050002.npy|sil zh uen1 zh ong4 k e1 x ve2 g uei1 l v4 d e5 ^ iao1 q iou2 sil
0|vits_data/waves-16k/SSB0005/SSB00050004.wav|vits_data/temps/SSB0005/SSB00050004.spec.pt|vits_data/berts/SSB0005/SSB00050004.npy|sil h ei1 k e4 x van1 b u4 zh iii3 ^ iao4 b o1 d a2 m ou3 ^ i2 g e4 d ian4 h ua4 sil
```
## 数据调试
```
python prep_debug.py
```

## 启动训练

```
cd monotonic_align

python setup.py build_ext --inplace

cd -

python train.py -c configs/bert_vits.json -m bert_vits
```

## 下载权重
AISHELL3_G.pth：https://github.com/PlayVoice/vits_chinese/releases/v4.0

## 推理测试
```
python vits_infer.py -c configs/bert_vits.json -m AISHELL3_G.pth -i 0
```
-i 为发音人序号，取值范围：0 ~ 173

**AISHELL3训练数据都是短短的一句话，所以，推理语句中不能有标点**

## 训练的AISHELL3模型，使用小米K2社区开源的AISHELL3模型来初始化训练权重，以节约训练时间

K2开源模型 https://huggingface.co/jackyqs/vits-aishell3-175-chinese/tree/main 下载模型

K2在线试用 https://huggingface.co/spaces/k2-fsa/text-to-speech
