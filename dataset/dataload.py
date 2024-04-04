import torch
from torch.utils.data import Dataset, DataLoader

from re import sub
import json
import librosa
import random
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

def collate_fn(batch):
    audio_feature, wav_path, caption, topk_audio_feature, topk_caption = zip(*batch)
    # retrieved results (확인)
    if topk_audio_feature[0]:
        topk_audio_feature = list(map(list, zip(*topk_audio_feature))) # B, top_k to top_k, B 와 이거였어!
        topk_caption = list(map(list, zip(*topk_caption)))
    else:
        topk_audio_feature = []
        topk_caption = []    
    return audio_feature, wav_path, caption, topk_audio_feature, topk_caption

def load_json_file(files, filtering = None):
    json_data = []

    if filtering:
        with open(filtering, 'r') as f:
            filtering = json.load(f)
    for file in files:
        with open(file, 'r') as f:
            json_obj = json.load(f)
            # id, caption, audio_path, duration, tag
            for item in json_obj['data']:
            # 여기서 freesound, audioset을 스킵?
                if ("FreeSound" in file and filtering) or ("AudioSet" in file and filtering):
                    continue
                if item['duration'] > 40.0:
                    continue
                json_data.append(item)
    return json_data

def load_dataset(data_path, filtering, train = None, index_path = None, top_k = 2):
    train_dataset = AudioLanguageDataset(data_path, filtering, train, index_path, top_k)
    return train_dataset

def load_dataloader(config, train_dataset, subset):
    # length 기반의 sampler 추가할지말지 나중에 결정.
    batch_size = 1 if 'val' in subset else config.data_args.batch_size
    num_workers = 2 if 'val' in subset else config.data_args.num_workers
    return DataLoader(
        dataset = train_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=config.data_args.shuffle,
        drop_last=False,
        collate_fn=collate_fn,
    )

def text_preprocess(sentence):
    def clean_text(text):
        text = text.lower()
        # ...
        text = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', text).replace('  ', ' ')
        # ...
        text = sub('[(,.!?;:|*\")]', ' ', text).replace('  ', ' ')
        return text

    if isinstance(sentence, str):
        return clean_text(sentence)
    elif isinstance(sentence, list):
        return [clean_text(s) for s in sentence]
    else:
        raise ValueError("Input should be a string or a list of strings.")
    

# https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/hook.py#L120 참고
class AudioLanguageDataset(Dataset):
    def __init__(self, json_files, filtering, train = True, index_path = None, topk = 2):
        # json file 불러오기
        self.json_data = load_json_file(json_files, filtering)
        
        # read retrieved result
        self.index_path = index_path
        if self.index_path:
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
                     
        # audio config 따르기
        self.audio_cfg = {
            "audio_length": 1024,
            "clip_samples": 480000,
            "mel_bins": 64,
            "sample_rate": 48000,
            "window_size": 1024,
            "hop_size": 480,
            "fmin": 50,
            "fmax": 14000,
            "class_num": 527,
            "model_type": "HTSAT",
            "model_name": "base"
        }     
        self.train = train
        self.topk = topk
       
    # clap preprocess 과정
    def preprocess_waveform(self, wav_path, duration):
        waveform, sr = librosa.load(wav_path, sr=self.audio_cfg['sample_rate'], duration=duration)
        audio_waveform = int16_to_float32(float32_to_int16(waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000,
            data_truncating='fusion',
            data_filling='repeatpad',
            audio_cfg=self.audio_cfg,
            require_grad=False,
        )
        return temp_dict
      
    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index):
        # index로 아이템 고르기
        items = self.json_data[index]
        
        # wav_path, caption, duration
        wav_path = items['audio']
        caption = items['caption']
        duration = items['duration']
        
        # preprocess: audio feature, caption, duration
        audio_feature = self.preprocess_waveform(wav_path, duration)
        caption = text_preprocess(caption)
        
        if self.train and isinstance(caption, list): # self.train and
            caption = random.choice(caption)
        
        topk_audio_feature = []
        topk_caption = []
        # retrieve context (topk도 설정할 수 있겠네.)
        if self.index_path != "":
            if wav_path in self.index:
                topk_result = self.index[wav_path]
                topk_results = topk_result[:self.topk] # topk, batch
                topk_audio_feature = [self.preprocess_waveform(wav_path, duration) for wav_path, _ in topk_results]
                topk_caption = [text_preprocess(caption) for _, caption in topk_results]
                # 각 순서별로 weights 5,4,3,2,1 ?
                # weights = list(range(len(top_k_result), 0, -1))
            else:
                print(wav_path)
        return audio_feature, wav_path, caption, topk_audio_feature, topk_caption
    
    