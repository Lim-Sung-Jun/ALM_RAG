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

def filtering_a2a():
    pass

def filtering_t2t():
    pass

def filtering_both():
    pass

def filtered_kb(saved_path):
    pass

def generate_filtered_kb(filtered_result):
    pass

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
    csv_path = os.path.join(base_path, f"{mode}_captions_wav_paths_{dimension}.csv")
    if modality == 'text':
        csv_path = os.path.join(base_path, "huge_sentenceTransformer_text_captions_wav_paths.csv")
    
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
    index_types = config.retrieval.index_types
    for modality in index_types:
        index_save_path = f"{save_path}/{modality}_faiss_index.bin"
        print(f"Faiss index for {modality} is ready with {selected_indices[modality].ntotal} vectors.")
        check_and_create_directory(save_path)
        faiss.write_index(selected_indices[modality], index_save_path)
        print(f"Faiss index for {modality} saved to {index_save_path}")

    print("Saved all selected indices")
    
if __name__ == "__main__":
    
    # 1. query: validation
    #
    # kb에서 a2a로 필터링하여 20개 추출
    # 20개를 t2t로 top1을 추출해서 5개 검색기록 만들기
    
    # kb에서 t2t로 필터링하여 top5개씩 20개 추출
    # 20개를 a2a로 top5를 추출해서 검색기록 만들기
    
    # kb에서 a2a와 t2t를 모두 고려하여 5개 추출
    # a2a와 각 caption의 top1을 추출해서 5개
    # a2a와 첫 번째 caption의 top5를 추출하여 5개

    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    
    # # 데이터 불러오기
    # val_dataset = load_dataset(config.val_dataset, config.filtering, False, config.retrieval.val_index_path, config.retrieval.topk) # config.retrieval.val_index_path
    
    # # 데이터로더 불러오기
    # val_dataloader = load_dataloader(config, val_dataset, 'val')
    
    # clotho_val로 a2a & t2t 검색
    # 1. precompute (text&audio embedding for query&kb) - done, # mode, modality, dimension
    base_path = "/data1/sungjun/index/data/"
    
    # audio
    query_embedding_clothoVal, captions_clothoVal, wav_paths_clothoVal = load_embedding(base_path, "val_512_embeddings_clotho", "val", "audio", 512)
    kb_embedding_wavcapsTrain, captions_wavcapsTrain, wav_paths_wavcapsTrain = load_embedding(base_path, "pretrain_512_embeddings", "pretrain", "audio", 512)
    kb_embedding_audiocapsTrain, captions_audiocapsTrain, wav_paths_audiocapsTrain = load_embedding(base_path, "train_512_embeddings_audiocaps", "train", "audio", 512)
    kb_embedding_clothoTrain, captions_clothoTrain, wav_paths_clothoTrain = load_embedding(base_path, "train_512_embeddings_clotho", "train", "audio", 512)
    # kb_embedding_autoacdTrain, captions_autoacdTrain, wav_paths_autoacdTrain = load_embedding(base_path, "train_512_embeddings_autoacd", "train", "audio", 512)
    
    # text
    # 5caption
    query_embedding_clothoVal_5text, captions_clothoVal_5text, wav_paths_clothoVal_5text = load_embedding(base_path, "val_sentenceTransformer_768_embeddings_clotho_5caption", "val", "text", 768)
    query_embedding_clothoVal_text, captions_clothoVal_text, wav_paths_clothoVal_text = load_embedding(base_path, "val_sentenceTransformer_768_embeddings_clotho", "val", "text", 768)
    
    kb_embedding_wavcapsTrain_text, captions_wavcapsTrain_text, wav_paths_wavcapsTrain_text = load_embedding(base_path, "pretrain_sentenceTransformer_768_embeddings_wavcaps", "pretrain", "text", 768)
    kb_embedding_audiocapsTrain_text, captions_audiocapsTrain_text, wav_paths_audiocapsTrain_text = load_embedding(base_path, "train_sentenceTransformer_768_embeddings_audiocaps", "train", "text", 768)
    kb_embedding_clothoTrain_text, captions_clothoTrain_text, wav_paths_clothoTrain_text = load_embedding(base_path, "train_sentenceTransformer_768_embeddings_clotho", "train", "text", 768)
    # kb_embedding_autoacdTrain_text, captions_autoacdTrain_text, wav_paths_autoacdTrain_text = load_embedding(base_path, "train_sentenceTransformer_768_embeddings_autoacd", "train", "text", 768)    
    
    def print_embeddings_info():
        # Audio data
        audio_data = {
            'clothoVal': (query_embedding_clothoVal, captions_clothoVal, wav_paths_clothoVal),
            'wavcapsTrain': (kb_embedding_wavcapsTrain, captions_wavcapsTrain, wav_paths_wavcapsTrain),
            'audiocapsTrain': (kb_embedding_audiocapsTrain, captions_audiocapsTrain, wav_paths_audiocapsTrain),
            'clothoTrain': (kb_embedding_clothoTrain, captions_clothoTrain, wav_paths_clothoTrain),
            # Uncomment or add more as needed
            # 'autoacdTrain': (kb_embedding_autoacdTrain, captions_autoacdTrain, wav_paths_autoacdTrain),
        }

        # Text data
        text_data = {
            'clothoVal_5text': (query_embedding_clothoVal_5text, captions_clothoVal_5text, wav_paths_clothoVal_5text),
            'clothoVal_text': (query_embedding_clothoVal_text, captions_clothoVal_text, wav_paths_clothoVal_text),
            'wavcapsTrain_text': (kb_embedding_wavcapsTrain_text, captions_wavcapsTrain_text, wav_paths_wavcapsTrain_text),
            'audiocapsTrain_text': (kb_embedding_audiocapsTrain_text, captions_audiocapsTrain_text, wav_paths_audiocapsTrain_text),
            'clothoTrain_text': (kb_embedding_clothoTrain_text, captions_clothoTrain_text, wav_paths_clothoTrain_text),
            # Uncomment or add more as needed
        }

        # Function to print info
        def print_info(data, modality):
            print(f"\n{modality} Embeddings and Info:")
            for name, (embedding, captions, wav_paths) in data.items():
                print(f"{name}_shape: {embedding.shape}, caption_length: {len(captions)}, wav_paths: {len(wav_paths)}")

        # Print for both modalities
        print_info(audio_data, "Audio")
        print_info(text_data, "Text")

    # Call the function to print the info
    print_embeddings_info()

    kb_large_audio = np.concatenate((kb_embedding_wavcapsTrain, kb_embedding_audiocapsTrain, kb_embedding_clothoTrain))
    kb_large_text = np.concatenate((kb_embedding_wavcapsTrain_text, kb_embedding_audiocapsTrain_text, kb_embedding_clothoTrain_text))
    kb_large_captions = pd.concat([captions_wavcapsTrain, captions_audiocapsTrain, captions_clothoTrain]).reset_index(drop = True)
    kb_large_wav_paths = pd.concat([wav_paths_wavcapsTrain, wav_paths_audiocapsTrain, wav_paths_clothoTrain]).reset_index(drop = True)
    # check1 - audio와 text의 caption, wav_path 길이랑 내용이 같은지 보기 & 임베딩의 길이 확인 및 shape 체크
    # check2 - dataset, embedding size 비교하기
    
    # 2. retrieve (a2a, t2t) - 동시에 고려한다.
    def make_index(embedding_dim):
        index_cpu = faiss.IndexFlatIP(embedding_dim)
        return index_cpu
    
    def create_indices(index_types, text_embedding_dim=768, audio_embedding_dim=512):
        indices = {}
        for index_type in index_types:
            if index_type == "text":
                indices["text"] = make_index(text_embedding_dim)
            elif index_type == "audio":
                indices["audio"] = make_index(audio_embedding_dim)
            elif index_type == "mixed":
                indices["mixed"] = make_index(audio_embedding_dim)
            elif index_type == "audio&text":
                indices["audio&text"] = make_index(audio_embedding_dim + text_embedding_dim + text_embedding_dim)
        return indices
    
    index_types = ["audio&text"]
    selected_indices = create_indices(index_types)
    ##################################################################################################################################################################################################################
    # audio kb
    kb_embeds_audio_np = np.array(kb_large_audio).copy().astype('float32')
    kb_embeds_text_np = np.array(kb_large_text).copy().astype('float32')
    kb_embeds_np = np.concatenate((kb_large_audio, kb_large_text, kb_large_text), axis = 1)
    faiss.normalize_L2(kb_embeds_np)
    selected_indices["audio&text"].add(kb_embeds_np)
    ##################################################################################################################################################################################################################
    # query
    # audio, text query embeddings
    query_audio_embeddings_np = np.array(query_embedding_clothoVal).copy().astype('float32')
    # query_text_embeddings_np = np.array(query_embedding_clothoVal_text).copy().astype('float32')
    # query_embeddings_np = np.concatenate((query_audio_embeddings_np, query_text_embeddings_np), axis = 1)
    # faiss.normalize_L2(query_embeddings_np)
    
    # audio, 5text query embeddings
    query_5text_embeddings_np = np.array(query_embedding_clothoVal_5text).copy().astype('float32')
    # query_5audio_embeddings_np = np.expand_dims(query_audio_embeddings_np, axis = 1).repeat(5, axis = 1)
    # query_5embeddings_np = np.concatenate((query_5audio_embeddings_np, query_5text_embeddings_np), axis = 2)
    # query_5embeddings_np = query_5embeddings_np.reshape(-1, query_5embeddings_np.shape[-1])
    # faiss.normalize_L2(query_5embeddings_np)
    
    # at2at (2text)
    first_query_5text_embeddings_np = query_5text_embeddings_np[:,0,:]
    second_query_5text_embeddings_np = query_5text_embeddings_np[:,1,:]
    query_2embeddings_np = np.concatenate((query_audio_embeddings_np, first_query_5text_embeddings_np, second_query_5text_embeddings_np), axis = 1)
    faiss.normalize_L2(query_2embeddings_np)
    
    ##################################################################################################################################################################################################################
    # at2at (caption avg)
    # 어떻게 구현하지. 이것만 고민해보자! concat을 해서 하면 되겠네 난 천재야 시발
    # D, I = selected_indices["audio&text"].search(query_embeddings_np, 12) # 이 갯수에 따라서 달라지려나?
    # create_json_file_retrieval(config, I, captions_clothoVal, wav_paths_clothoVal, kb_large_captions, kb_large_wav_paths, "audio&text")
    # save_index(selected_indices, config)
    
    # at2at (caption 5)
    # D, I = selected_indices["audio&text"].search(query_5embeddings_np, 2)
    # create_json_file_retrieval(config, I, captions_clothoVal, wav_paths_clothoVal, kb_large_captions, kb_large_wav_paths, "audio&text_5caption")
    # save_index(selected_indices, config)
    
    D, I = selected_indices["audio&text"].search(query_2embeddings_np, 12)
    create_json_file_retrieval(config, I, captions_clothoVal, wav_paths_clothoVal, kb_large_captions, kb_large_wav_paths, "audio&text_2caption")
    save_index(selected_indices, config)
    
    # 필터링된 kb에서 a2a 검색하여 retrieved result 만들기
    # - filtered_kb의 embedding precompute
    # - clotho_val 임베딩, kb 임베딩 -> search
    
    print()