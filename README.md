### VITS实现的中文TTS，集成部分微软NaturalSpeech优化措施

this is the copy of https://github.com/jaywalnut310/vits		

VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech		

Espnet连接：github.com/espnet/espnet/tree/master/espnet2/gan_tts/vits

coqui-ai/TTS连接：github.com/coqui-ai/TTS/tree/main/recipes/ljspeech/vits_tts


### 基于VITS 实现 16K TTS 的流程记录

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

### Data Link
https://github.com/PlayVoice/HuaYan_TTS

### 将16K音频拷贝到./baker_waves/，启动训练

python train.py -c configs/baker_base.json -m baker_base

两张1080卡，训练两天，基本可以使用了

![LOSS值](/configs/loss.png)

### 测试
python vits_strings.py

### iSTFT
完成
