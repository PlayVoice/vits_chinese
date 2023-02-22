### Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft

based on BERT，NaturalSpeech, VITS

### Infer

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

#### Down Pretrained model

BaiduYun：https://pan.baidu.com/s/1Cj4MnwFyZ0XZmTR6EpygbQ?pwd=yn60

Or down from release

prosody_model.pt To ./bert/prosody_model.pt

vits_bert.pth To ./vits_bert.pth

python vits_infer.py

./vits_infer_out have the waves infered

### Train
going

### other data Link
https://github.com/PlayVoice/HuaYan_TTS


