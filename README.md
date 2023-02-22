### Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

Based on BERT, NaturalSpeech, VITS

### Features
1, Hidden prosody embedding from BERT，自然的停顿

2, Infer loss from NaturalSpeech，超低的发音错误

3, Framework of VITS，超好的音质

### Install

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

### Infer with Pretrained model

BaiduYun：https://pan.baidu.com/s/1Cj4MnwFyZ0XZmTR6EpygbQ?pwd=yn60

Or down from release page

put prosody_model.pt To ./bert/prosody_model.pt

put vits_bert.pth To ./vits_bert.pth

python vits_infer.py

./vits_infer_out have the waves infered

### Train
going

### other data Link
https://github.com/PlayVoice/HuaYan_TTS


