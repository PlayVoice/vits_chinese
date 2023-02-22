### Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

Based on BERT, NaturalSpeech, VITS

### Features
1, Hidden prosody embedding from BERT，get natural pauses in grammar

2, Infer loss from NaturalSpeech，get less sound error

3, Framework of VITS，get high audio quality

### Install

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

### Infer with Pretrained model

BaiduYun：https://pan.baidu.com/s/1Cj4MnwFyZ0XZmTR6EpygbQ?pwd=yn60

Google: https://drive.google.com/drive/folders/1sioiNpebOLyCmHURgOgJ7ppWI7b-7Rb5?usp=sharing

Or get from release page

put prosody_model.pt To ./bert/prosody_model.pt

put vits_bert_model.pth To ./vits_bert_model.pth

python vits_infer.py

./vits_infer_out have the waves infered

### Train
going

### Other data Link
https://github.com/PlayVoice/HuaYan_TTS


