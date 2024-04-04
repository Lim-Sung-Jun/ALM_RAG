import types
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
import faiss
import torch
import pandas as pd
import time
import os
import librosa
from laion_clap import CLAP_Module
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import torch.nn.functional as F
def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, default="./configs/pretrain.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config

def create_json_file_retrieval(config, I, query_modality, kb_modality, query_captions, query_wav_paths, kb_captions, kb_wav_paths, x): 
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
    check_and_create_directory(config.index_args.index_save_path)
        
    with open(f'{config.index_args.index_save_path}/text2{x}.json', 'w') as file:
        json.dump(results, file, indent=4)

def check_and_create_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")
        
def save_index(selected_indices, config):
    save_path = config.index_args.index_save_path
    index_types = config.index_args.index_types
    for modality in index_types:
        index_save_path = f"{save_path}/{modality}_faiss_index.bin"
        print(f"Faiss index for {modality} is ready with {selected_indices[modality].ntotal} vectors.")
        check_and_create_directory(save_path)
        faiss.write_index(selected_indices[modality], index_save_path)
        print(f"Faiss index for {modality} saved to {index_save_path}")

    print("Saved all selected indices")

def load_embeddings_and_csv(base_path):
    """
    Load embeddings using memory mapping and CSV files from the specified directory.
    """
    # File path for embeddings
    embeddings_path = os.path.join(base_path, "huge_sentenceTransformer_text_embeddings.npy")
    csv_path = os.path.join(base_path, f"huge_sentenceTransformer_text_captions_wav_paths.csv")
    # Load embeddings using memory mapping
    embeddings = np.load(embeddings_path, mmap_mode='r')
    # Load CSV file
    df = pd.read_csv(csv_path)
    captions, wav_paths = df['caption'], df['wav_path']

    return embeddings, captions, wav_paths
def filtering_score(D, I, threshold = 0.5):

    # Filter the results using list comprehension
    filtered_D = [d[d < threshold] for d in D]
    filtered_I = [i[d < threshold] for i, d in zip(I, D)]

    # Convert lists to arrays of objects to handle varying lengths
    filtered_D = np.array(filtered_D, dtype=object)
    filtered_I = np.array(filtered_I, dtype=object)
    return filtered_D[:,:7], filtered_I[:,:7]

if __name__ == "__main__":
    config = get_config()
    print(config)
    knowledge_base_path = ["./data/train_sentenceTransformer_768_embeddings_audiocaps", "./data/train_sentenceTransformer_768_embeddings_clotho", "data/pretrain_sentenceTransformer_768_embeddings_wavcaps"] # "data/train_sentenceTransformer_768_embeddings_autoacd"
    query_path = ["./data/clotho_valid_text2text"] # + validation set #"./data/val_512_embeddings_macs", "./data/val_768_embeddings_macs"]
    
    csv_path = '/home/sungjun/projects/audio_rag/AudioRAG/data/clotho_valid_text2text_filtered_kb/val_captions_wav_paths_512.csv'
    audio_embedding_path = '/home/sungjun/projects/audio_rag/AudioRAG/data/clotho_valid_text2text_filtered_kb/val_audio_whole_512_embeddings.npy'
    text_embedding_path = '/home/sungjun/projects/audio_rag/AudioRAG/data/clotho_valid_text2text_filtered_kb/val_text_whole_512_embeddings.npy'
    kb_audio_embeddings = np.load(audio_embedding_path, mmap_mode = 'r')
    kb_text_embeddings = np.load(text_embedding_path, mmap_mode = 'r')
    df = pd.read_csv(csv_path)
    kb_captions, kb_wav_paths = df['caption'], df['wav_path']
    
    csv_path = '/home/sungjun/projects/audio_rag/AudioRAG/data/val_512_embeddings_clotho/val_captions_wav_paths_512.csv'
    audio_embedding_path = '/home/sungjun/projects/audio_rag/AudioRAG/data/val_512_embeddings_clotho/val_audio_512_embeddings.npy'
    clotho_val = np.load(audio_embedding_path, mmap_mode = 'r')
    df = pd.read_csv(csv_path)
    clotho_val_captions, clotho_val_wav_paths = df['caption'], df['wav_path']
    
    # clotho_val, clotho_val_captions, clotho_val_wav_paths = load_embeddings_and_csv(query_path[0])
    clotho_train, clotho_captions, clotho_wav_paths = load_embeddings_and_csv(knowledge_base_path[1])
    audiocaps_train, audiocaps_captions, audiocaps_wav_paths = load_embeddings_and_csv(knowledge_base_path[0])
    wavcaps_train, wavcaps_captions, wavcaps_wav_paths = load_embeddings_and_csv(knowledge_base_path[2])  
    # autoacd_train, autoacd_captions, autoacd_wav_paths = load_embeddings_and_csv(knowledge_base_path[3])
    
    clotho_val = clotho_val.astype('float32')
    clotho_train = clotho_train.astype('float32')
    audiocaps_train = audiocaps_train.astype('float32')
    wavcaps_train = wavcaps_train.astype('float32')
    
    # clotho_val_embeddings = clotho_val_embeddings.astype('float32')
    kb_audio_embeddings = kb_audio_embeddings.astype('float32')
    kb_text_embeddings = kb_text_embeddings.astype('float32')
    # autoacd_train = autoacd_train.astype('float32')
  
    # kb: wavcaps, audioset, clotho (train + pretrain)
    if config.index_args.big_kb:
        # large      
        kb_large_text = np.concatenate((audiocaps_train, clotho_train, wavcaps_train))
        kb_large_captions = pd.concat([audiocaps_captions, clotho_captions, wavcaps_captions]).reset_index(drop = True)
        kb_large_wav_paths = pd.concat([audiocaps_wav_paths, clotho_wav_paths, wavcaps_wav_paths]).reset_index(drop = True)
        
    query_captions = {
                    'clotho_train_text': clotho_captions,
                    'clotho_val': clotho_val_captions
                    }
    query_wav_paths = {
                    'clotho_train_text': clotho_wav_paths,
                    'clotho_val': clotho_val_wav_paths
                    }

    def make_index(embedding_dim, nlist):
        index_cpu = faiss.IndexFlatIP(embedding_dim)
        return index_cpu
    
    def create_indices(index_types, text_embedding_dim=512, audio_embedding_dim=512, nlist=128):
        indices = {}

        for index_type in index_types:
            if index_type == "text":
                indices["text"] = make_index(text_embedding_dim, nlist)
            elif index_type == "audio":
                indices["audio"] = make_index(audio_embedding_dim, nlist)
            elif index_type == "frame":
                indices["frame"] = make_index(audio_embedding_dim, nlist)
            elif index_type == "mixed":
                indices["mixed"] = make_index(audio_embedding_dim, nlist)
            else:
                print(f"Invalid index type: {index_type}")
                indices[index_type] = make_index(audio_embedding_dim, nlist)

        return indices
    
    # audio, text, frame index
    index_types = ["audio", "text"]
    selected_indices = create_indices(index_types)
    
    query_list = [clotho_val] # query_val_audio_clotho_embeddings

    # # audio
    kb_embeds_audio_np = np.array(kb_audio_embeddings).copy().astype('float32')
    faiss.normalize_L2(kb_embeds_audio_np)
    selected_indices["audio"].add(kb_embeds_audio_np)
    
    # text
    kb_embeds_text_np = np.array(kb_text_embeddings).copy().astype('float32')
    faiss.normalize_L2(kb_embeds_text_np)
    selected_indices["text"].add(kb_embeds_text_np)

    kb_caption = {
                    'large_text': kb_large_captions,
                    'filtered_kb': kb_captions
                }
    kb_wav_path = {
                    'large_text': kb_large_wav_paths,
                    'filtered_kb': kb_wav_paths
                }
    
    query_audio_embeddings = query_list[0] # audiocaps_val, clotho_val
    query_audio_embeddings_np = np.array(query_audio_embeddings).copy().astype('float32')
    faiss.normalize_L2(query_audio_embeddings_np) # B, 512
    D, I = selected_indices["audio"].search(query_audio_embeddings_np, 7)
    create_json_file_retrieval(config, I, "clotho_val", "filtered_kb", query_captions["clotho_val"], query_wav_paths["clotho_val"], kb_caption["filtered_kb"], kb_wav_path["filtered_kb"], "audio")
    D, I = selected_indices["text"].search(query_audio_embeddings_np, 7)
    create_json_file_retrieval(config, I, "clotho_val", "filtered_kb", query_captions["clotho_val"], query_wav_paths["clotho_val"], kb_caption["filtered_kb"], kb_wav_path["filtered_kb"], "text")
    
    save_index(selected_indices, config)
    