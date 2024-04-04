import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
import json
import sys
# Calculate the path to the parent directory
script_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
parent_directory = os.path.dirname(script_directory)  # Parent directory

# Add the parent directory to sys.path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)
from utils.utils import get_config
from dataset.dataload import load_dataset, load_dataloader
from omegaconf import OmegaConf

def get_config(args):
    # 파일에서 config 가져오기에 좋다.
    config = OmegaConf.load(args.config)
    return config

def load_embedding(base_path, embedding_path, mode, modality, dimension):
    base_path = base_path + embedding_path
    
    embeddings_path = os.path.join(base_path, f"{mode}_{modality}_{dimension}_embeddings.npy")
    if modality == 'text':
        embeddings_path = os.path.join(base_path, "huge_sentenceTransformer_text_embeddings.npy")
    if modality == 'filter':
        embeddings_path = os.path.join(base_path, "filtered_audio_embeddings.npy")
    csv_path = os.path.join(base_path, f"{mode}_captions_wav_paths_{dimension}.csv")
    if modality == 'text':
        csv_path = os.path.join(base_path, "huge_sentenceTransformer_text_captions_wav_paths.csv")
    if modality == 'filter':
        csv_path = os.path.join(base_path, "filtered_captions_wav_paths_512.csv")
    
    embeddings = np.load(embeddings_path, mmap_mode = 'r').astype('float32')
    df = pd.read_csv(csv_path)
    captions, wav_paths = df['caption'], df['wav_path']
    
    return embeddings, captions, wav_paths

def create_json_file_retrieval(config, I, query_captions, query_wav_paths, kb_captions, kb_wav_paths, x): 
    results = {}
    print('json files save...')
    start_time = time.time()

    # my code
    for query_index, topk_indices in tqdm(enumerate(I)):
        query_audio_path = query_wav_paths[query_index]
        query_caption = query_captions[query_index]
        
        query = (query_audio_path, query_caption)

        topk_items = []
        for i in topk_indices:
            kb_item = (kb_wav_paths[i], kb_captions[i])
            if kb_item != query:
                topk_items.append(kb_item)
            if len(topk_items) == 10:
                break

        results[query_audio_path] = topk_items  

    elapsed_time = time.time() - start_time
    print(f"elapsed time for saving json files_text2{x}: {elapsed_time}")
    print('json files save complete...')

    # Assuming check_and_create_directory is a function you have defined elsewhere
    check_and_create_directory(config.retrieval.index_save_path)
        
    with open(f'{config.retrieval.index_save_path}{x}2{x}.json', 'w') as file:
        json.dump(results, file, indent=4)

def check_and_create_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")
        
def save_index(selected_indices, config):
    save_path = config.retrieval.index_save_path
    index_save_path = f"{save_path}/faiss_index.bin"
    print(f"Faiss index for filtered is ready with {selected_indices.ntotal} vectors.")
    check_and_create_directory(save_path)
    faiss.write_index(selected_indices, index_save_path)
    print(f"Faiss index for filtered saved to {index_save_path}")

    print("Saved all selected indices")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    base_path = "/data1/sungjun/index/data/"
    
    # audio
    query_embedding_clothoVal, captions_clothoVal, wav_paths_clothoVal = load_embedding(base_path, "val_512_embeddings_clotho", "val", "audio", 512)
    kb_embedding, captions, wav_paths = load_embedding("./filtered_data/", "", "", "filter", 512)
    
    print(query_embedding_clothoVal.shape, len(captions_clothoVal), len(wav_paths_clothoVal))
    print(kb_embedding.shape, len(captions), len(wav_paths))
    
    index_cpu = faiss.IndexFlatIP(512)
    
    faiss.normalize_L2(kb_embedding)
    index_cpu.add(kb_embedding)
    
    query_audio_embeddings_np = np.array(query_embedding_clothoVal).copy().astype('float32')
    faiss.normalize_L2(query_audio_embeddings_np)
    
    D, I = index_cpu.search(query_audio_embeddings_np, 8)
    create_json_file_retrieval(config, I, captions_clothoVal, wav_paths_clothoVal, captions, wav_paths, "audio2audio")
    save_index(index_cpu, config)
    
    print()