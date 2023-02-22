### Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

Based on BERT, NaturalSpeech, VITS

### Features
1, Hidden prosody embedding for BERT

2, Infer loss add from NaturalSpeech

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


