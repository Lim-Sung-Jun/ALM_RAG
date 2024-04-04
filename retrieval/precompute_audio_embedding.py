import sys
import os
import argparse
import numpy as np
import pandas as pd
# Calculate the path to the parent directory
script_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
parent_directory = os.path.dirname(script_directory)  # Parent directory

# Add the parent directory to sys.path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)
from utils.utils import get_config
from dataset.dataload import load_dataset, load_dataloader
from omegaconf import OmegaConf
import torch
import time
from tqdm import tqdm
from laion_clap import CLAP_Module

def check_and_create_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")

def process_audio_data(data, clap_model, config):
    # if config.index_args.audio_dimension == 768: # before projection
    #     def get_audio_embedding_before_projection(self, data):
    #             """Get the audio embedding from the model

    #             Parameters
    #             ----------
    #             data: a list of dict
    #                 the audio input dict list from 'get_audio_feature' method

    #             Returns
    #             ----------
    #             audio_embed: torch.Tensor
    #                 a tensor of audio_embeds (N, D)

    #             """
    #             device = next(self.parameters()).device
    #             input_dict = {}
    #             keys = data[0].keys()
    #             for k in keys:
    #                 input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
    #             audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
    #             return audio_embeds
    #     clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model)
    #     # 패딩처리 추가하기 아래 함수에서 data는 N, T를 입력받음.    
    return clap_model.get_audio_embedding_from_data(x=data, use_tensor=True)

if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    
    # dataload
    val_dataset = load_dataset(config.fitered_dataset, config.filtering, False, config.retrieval.val_index_path, config.retrieval.topk) # config.retrieval.val_index_path
    val_dataloader = load_dataloader(config, val_dataset, 'filtered')
    
    # model
    # clap
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap = CLAP_Module(enable_fusion=True, device=device)  # 615M
    clap.load_ckpt()
    
    captions = []
    wav_paths = []
    embeddings = []
    from itertools import islice
    sliced_val_dataloader = islice(val_dataloader, 10)
    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(val_dataloader):
            audio, audio_path, caption, topk_audio, topk_caption = batch
            captions.extend(caption)
            wav_paths.append(audio_path)
            data = [item['waveform'] for item in audio] # wavform_data = [item['waveform'] for temp in data for item in temp]
            outputs = process_audio_data(data, clap, config) # audio # 리스트를 잘 처리해서 다차원으로 만든다.
            embeddings.extend(outputs.cpu().contiguous())

    wav_paths_list = [item for sublist in wav_paths for item in sublist]
    
    # Save the embeddings, captions, wav_paths_list
    save_path = config.retrieval.index_save_path
    os.makedirs(save_path, exist_ok=True)
    
    embeddings_array = torch.stack(embeddings).numpy()
    embedding_save_path = os.path.join(save_path, f"filtered_audio_embeddings.npy")

    np.save(embedding_save_path, embeddings_array)
    print(f"Saved embeddings to {embedding_save_path}")

    captions_df = pd.DataFrame({
        'caption': captions,
        'wav_path': wav_paths_list
    })
    captions_csv_path = os.path.join(save_path, f"filtered_captions_wav_paths_512.csv")
    captions_df.to_csv(captions_csv_path, index=False)
    print(f"Captions and wav paths saved to {captions_csv_path}")
