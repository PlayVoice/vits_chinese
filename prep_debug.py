import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

from tqdm import tqdm
from data_utils import TextAudioLoader

hps = utils.get_hparams_from_file("configs/bert_vits.json")
dataset = TextAudioLoader("filelists/all.txt", hps.data, debug=True)

for _ in tqdm(dataset):
    pass
