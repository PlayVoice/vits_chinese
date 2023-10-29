## 数据下载
http://www.openslr.org/93/

## 采样率转换
```
python prep_resample.py --wav aishell-3/train/wav/ --out vits_data/waves-16k
```

## 标注规范化（lables.txt，名称不能改）
```
python prep_format_label.py --txt aishell-3/train/content.txt --out vits_data/lables.txt
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
python vits_infer.py -c configs/bert_vits.json -m AISHELL3_G.pth -i 6
```
-i 为发音人序号，取值范围：0 ~ 173

## 训练的AISHELL3模型，使用小米K2社区开源的AISHELL3模型来初始化训练权重，以节约训练时间
K2开源模型 https://huggingface.co/jackyqs/vits-aishell3-175-chinese/tree/main 下载模型

K2在线试用 https://huggingface.co/spaces/k2-fsa/text-to-speech
