vits实现的中文TTS

this is the copy of https://github.com/jaywalnut310/vits		

VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech		

Espnet连接：github.com/espnet/espnet/tree/master/espnet2/gan_tts/vits

coqui-ai/TTS连接：github.com/coqui-ai/TTS/tree/main/recipes/ljspeech/vits_tts

base on:https://github.com/lutianxiong/vits_chinese

如果有侵权行为，请联系我，我将删除项目

If there is infringement, please contact me and I will delete the item

# 基于VITS 实现 16K baker TTS 的流程记录

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

# 将16K标贝音频拷贝到./baker_waves/，启动训练

python train.py -c configs/baker_base.json -m baker_base

两张1080卡，训练两天，基本可以使用了

# 测试
python vits_strings.py

