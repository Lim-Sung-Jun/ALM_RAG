from sentence_transformers import SentenceTransformer
import json
import os
import faiss
import numpy as np
import pandas as pd
import torch
import argparse
import librosa
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn.functional as F

import sys
# Calculate the path to the parent directory
script_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
parent_directory = os.path.dirname(script_directory)  # Parent directory

# Add the parent directory to sys.path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)
from utils.utils import get_config
from dataset.dataload import load_dataset, load_dataloader

if __name__ == "__main__":
    name = "huge_sentenceTransformer_text"

    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    
    dataset = load_dataset(config.pretrain_dataset, config.filtering, False, config.retrieval.val_index_path, config.retrieval.topk) # config.retrieval.val_index_path
    
    # 데이터로더 불러오기
    dataloader = load_dataloader(config, dataset, 'pretrain')
    # dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0, seperation = config.index_args.seperation, mode = query_mode, segment_duration = 10)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    # Load pre-trained model
    text_encoder = SentenceTransformer("all-mpnet-base-v2")
    text_encoder = text_encoder.to(device)

    name_map = {
        'audio_sample': 0,
        'caption': 1,
        'wav_path': 2,
        'tags': 5
    }
    captions = []
    wav_paths = []
    embeddings = []
    from itertools import islice
    sample = 5
    # mode에 따라서 dataloader 선택
    with torch.no_grad(): # avg로도 해보자!
        for batch in tqdm(dataloader):# batch -> audio_features, caption, wav_paths, retr_audio_features, retr_captions, tags
            first_caption = [row[0] for row in batch[name_map["caption"]]]
            captions.extend(first_caption)
            wav_paths.append(batch[name_map["wav_path"]])
            data = batch[name_map['caption']]
            # avg
            document_embedding = []
            for i, _ in enumerate(data):
                document_embeddings = text_encoder.encode(data[i], convert_to_tensor=True).mean(0)
                document_embedding.append(document_embeddings)
            # single
            document_embedding = torch.stack(document_embedding)
            # document_embedding = text_encoder.encode(data, convert_to_tensor=True)
            embeddings.extend(document_embedding.cpu().contiguous())

    wav_paths_list = [item for sublist in wav_paths for item in sublist]
    
    # Save the embeddings, captions, wav_paths_list
    save_path = config.index_args.index_save_path
    os.makedirs(save_path, exist_ok=True)

    embeddings_array = torch.stack(embeddings).numpy()
    embedding_save_path = os.path.join(save_path, f"{name}_embeddings.npy")
    np.save(embedding_save_path, embeddings_array)
    print(f"Saved embeddings to {embedding_save_path}")
            
    captions_df = pd.DataFrame({
            'caption': captions,
            'wav_path': wav_paths_list
    })
         
    captions_csv_path = os.path.join(save_path, f"{name}_captions_wav_paths.csv")
    captions_df.to_csv(captions_csv_path, index=False)
    print(f"Captions and wav paths saved to {captions_csv_path}")

            