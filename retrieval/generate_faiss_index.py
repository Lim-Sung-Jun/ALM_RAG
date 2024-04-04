# Preprocess Audio file and Compute Embeddings
# Build retrieval database : Used for retrieving neighbors
# Build index for similarity search : Train and build a search index for querying neighbors.
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
# from data_handling.retrieval_dataset import RetrievalIndex
# from data_handling.pretrain_dataset import pretrain_dataloader

def check_and_create_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")

def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, default="./configs/pretrain.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config

def process_text_data(data, clap_model, config):
    return clap_model.get_text_embedding(x=data, use_tensor=True)

def process_audio_data(data, clap_model, config):
    if config.index_args.audio_dimension == 768: # before projection
        def get_audio_embedding_before_projection(self, data):
                """Get the audio embedding from the model

                Parameters
                ----------
                data: a list of dict
                    the audio input dict list from 'get_audio_feature' method

                Returns
                ----------
                audio_embed: torch.Tensor
                    a tensor of audio_embeds (N, D)

                """
                device = next(self.parameters()).device
                input_dict = {}
                keys = data[0].keys()
                for k in keys:
                    input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
                audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
                return audio_embeds
        clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model)
        # 패딩처리 추가하기 아래 함수에서 data는 N, T를 입력받음.
    if config.index_args.seperation: # 좋은 코드인지는 모르겠다..
        max_size = 7 # audio and tag audio
        element_list = []
        for element in data:
            element_list.append(clap_model.get_audio_embedding_from_data(x=element, use_tensor=True))
        padded_tensors = [F.pad(tensor, (0, 0, 0, max_size - tensor.size(0))) for tensor in element_list]
        stacked_tensor = torch.stack(padded_tensors)

        return stacked_tensor

    return clap_model.get_audio_embedding_from_data(x=data, use_tensor=True)

def process_frame_data(data, clap_model, config):
    if config.index_args.audio_dimension == 768: # before projection
        def get_audio_embedding_before_projection(self, data):
                """Get the audio embedding from the model

                Parameters
                ----------
                data: a list of dict
                    the audio input dict list from 'get_audio_feature' method

                Returns
                ----------
                audio_embed: torch.Tensor
                    a tensor of audio_embeds (N, D)

                """
                device = next(self.parameters()).device
                input_dict = {}
                keys = data[0].keys()
                for k in keys:
                    input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
                audio_embeds = self.encode_audio(input_dict, device=device)["fine_grained_embedding"]
                return audio_embeds
        clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model)
        audio_embed = clap_model.get_audio_embedding_from_data(x=data, use_tensor=True)
        chunks = audio_embed.chunk(4, dim=1)
        averaged_chunks = [chunk.mean(dim=1, keepdim=True) for chunk in chunks]
        audio_embeds = torch.cat(averaged_chunks, dim=1) # B, S, 768 -> EX. 2, 4, 768
    return audio_embeds

def precompute_and_save(config, dataloader, device, mode):
    # seperate 오디오 관련 파일
    # audiocaps
        # /drl_nas1/ckddls1321/data/AudioCaps/waveforms/train
        # /drl_nas1/ckddls1321/data/AudioCaps/waveforms/val
    # clotho
        # /drl_nas1/ckddls1321/data/CLOTHO_v2.1/clotho_audio_files/validation
        # /drl_nas1/ckddls1321/data/CLOTHO_v2.1/clotho_audio_files/development
    # auto-acd
        # /drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/Auto_ACD/train.json
        # /drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/Auto_ACD/test.json
    captions = []
    wav_paths = []
    tags = []
    
    index_types = config.index_args.index_types
    
    # model
    clap = CLAP_Module(enable_fusion=True, device = device)  # 615M
    clap.load_ckpt()  # download the default pretrained checkpoint.

    modalities = {
        "text": {"process": process_text_data, "embeddings": [], "data_key": "caption"}, # text embeddings
        "audio": {"process": process_audio_data, "frame_process": process_frame_data, "embeddings": [], "frame_embeddings": [], "data_key": "audio_sample"} # audio embeddings
        # 새로운 모달을 추가할 수 있다.
    }
    
    # from itertools import islice
    # num_batches_to_test = 5
    # tqdm(islice(dataloader, num_batches_to_test)):
    # tqdm(dataloader):
    
    name_map = {
        'audio_sample': 0,
        'caption': 1,
        'wav_path': 2,
        'tags': 5
    }
     
    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(dataloader):# batch -> audio_features, caption, wav_paths, retr_audio_features, retr_captions, tags
            captions.extend(batch[name_map["caption"]])
            wav_paths.append(batch[name_map["wav_path"]])
            if config.index_args.seperation:
                tags.append(batch[name_map["tags"]])
            for modality, info in modalities.items():
                if modality in index_types:
                    data = batch[name_map[info["data_key"]]]
                    if modality == 'audio':
                        if config.index_args.seperation:
                            data = [[item['waveform'] for item in element] for element in data]
                            # mel_fusion, longer, waveform 중에서 waveform만 가져오는 과정.
                        else:
                            data = [item['waveform'] for item in data]
                    outputs = info["process"](data, clap, config) # audio
                    # frame 코드
                    if modality == 'audio' and config.index_args.audio_dimension == 768:
                        frame_outputs = info["frame_process"](data, clap, config) 
                        info["frame_embeddings"].extend(frame_outputs.cpu().contiguous())
                    info["embeddings"].extend(outputs.cpu().contiguous())

    wav_paths_list = [subsublist[0] for sublist in wav_paths for subsublist in sublist]
    if config.index_args.seperation:
        tags_list = [tag for tag_list in tags for tag in tag_list]
    
    # Save the embeddings, captions, wav_paths_list
    save_path = config.index_args.index_save_path
    os.makedirs(save_path, exist_ok=True)

    # Dimension information
    audio_dim = config.index_args.audio_dimension
    text_dim = config.index_args.text_dimension
        
    # Save modalities (embeddings)
    for modality, info in modalities.items():
        if modality in index_types:
            # # Determine the dimension based on modality
            # dim = audio_dim if modality == 'audio' else text_dim

            # # Save standard embeddings
            # embedding_save_path = os.path.join(save_path, f"{mode}_{modality}_{dim}_embeddings.pkl")
            # with open(embedding_save_path, 'wb') as file:
            #     pickle.dump(info["embeddings"], file)
            # print(f"Saved {modality} embeddings to {embedding_save_path}")

            # Additionally, save frame embeddings for 'audio' in 'frame2audio' mode
            # if modality == 'audio' and audio_dim == 768:
            #     frame_embedding_save_path = os.path.join(save_path, f"{mode}_{modality}_{dim}_frame_embeddings.pkl")
            #     with open(frame_embedding_save_path, 'wb') as file:
            #         pickle.dump(info["frame_embeddings"], file)
            #     print(f"Saved {modality} frame embeddings to {frame_embedding_save_path}")
            
            # Convert embeddings to NumPy array
            embeddings_array = torch.stack(info["embeddings"]).numpy()

            # Determine the dimension based on modality
            dim = audio_dim if modality == 'audio' else text_dim

            # Save standard embeddings as .npy
            embedding_save_path = os.path.join(save_path, f"{mode}_{modality}_{dim}_embeddings.npy")
            np.save(embedding_save_path, embeddings_array)
            print(f"Saved {modality} embeddings to {embedding_save_path}")
            
            if modality == 'audio' and audio_dim == 768:
                frame_embeddings_array = torch.stack(info["frame_embeddings"]).numpy()
                frame_embedding_save_path = os.path.join(save_path, f"{mode}_{modality}_{dim}_frame_embeddings.npy")
                np.save(frame_embedding_save_path, frame_embeddings_array)
                print(f"Saved {modality} frame embeddings to {frame_embedding_save_path}")
    if captions and wav_paths_list and config.index_args.seperation:
        captions_df = pd.DataFrame({
            'caption': captions,
            'wav_path': wav_paths_list,
            'tags': tags_list
        })
    elif captions and wav_paths_list:
        captions_df = pd.DataFrame({
            'caption': captions,
            'wav_path': wav_paths_list
        })
         
    captions_csv_path = os.path.join(save_path, f"{mode}_captions_wav_paths_{audio_dim}.csv")
    captions_df.to_csv(captions_csv_path, index=False)
    print(f"Captions and wav paths saved to {captions_csv_path}")

    return modalities, captions, wav_paths_list


# def load_embeddings_and_csv(base_path, mode, modality, dimension, frame = None):
#     """
#     Load embeddings and CSV files from the specified directory.

#     Parameters:
#     - base_path: Base directory where files are stored.
#     - mode: 'pretrain' or 'train'.
#     - modality: 'audio' or 'text'.
#     - dimension: Dimension of the embeddings (e.g., 512).

#     Returns:
#     - embeddings: Loaded embeddings from the .pkl file.
#     - captions_df: DataFrame loaded from the .csv file.
#     """
#     # Construct file paths {mode}_{modality}_{dim}_frame_embeddings.pkl
#     if frame == 'frame':
#         embeddings_path = os.path.join(base_path, f"{mode}_{modality}_{dimension}_frame_embeddings.pkl")
#     else:
#         embeddings_path = os.path.join(base_path, f"{mode}_{modality}_{dimension}_embeddings.pkl")
#     csv_path = os.path.join(base_path, f"{mode}_captions_wav_paths_{dimension}.csv")

#     # Load embeddings
#     with open(embeddings_path, 'rb') as file:
#         embeddings = pickle.load(file)

#     # Load CSV file
#     df = pd.read_csv(csv_path)
    
#     captions, wav_paths = df['caption'], df['wav_path']

#     return embeddings, captions, wav_paths

def load_embeddings_and_csv(base_path, mode, modality, dimension, frame=None, sep = None):
    """
    Load embeddings using memory mapping and CSV files from the specified directory.
    """
    # File path for embeddings
    if frame == 'frame':
        embeddings_path = os.path.join(base_path, f"{mode}_{modality}_{dimension}_frame_embeddings.npy")
    else:
        embeddings_path = os.path.join(base_path, f"{mode}_{modality}_{dimension}_embeddings.npy")

    # File path for CSV
    csv_path = os.path.join(base_path, f"{mode}_captions_wav_paths_{dimension}.csv")

    # Load embeddings using memory mapping
    embeddings = np.load(embeddings_path, mmap_mode='r')

    # Load CSV file
    df = pd.read_csv(csv_path)
    if sep:
        captions, wav_paths, tags = df['caption'], df['wav_path'], df['tags']
        return embeddings, captions, wav_paths, tags
    else:
        captions, wav_paths = df['caption'], df['wav_path']

    return embeddings, captions, wav_paths


# def convert_pkl_to_mmap():
def make_retrieval_result_jsonfiles(config, query_path, knowledge_base_path): #
    retrieved_result = []

    # query_mode: ["audio2audio","audio2text","frame2audio"]
    query_mode = config.index_args.query_mode
    
    # query: audio512, frame768
    print('data loading...')
    query_audio_embeddings, train_captions, train_wav_paths = load_embeddings_and_csv(query_path[0], "train", "audio", 512)
    #query_frame_embeddings, _, _ = load_embeddings_and_csv(query_path[1], "train", "audio", 768, 'frame') # frame은 어떻게 저장되지?
    query_val_embeddings, val_captions, val_wav_paths = load_embeddings_and_csv(query_path[4], "val", "audio", 512) #
    # query_val_frame_embeddings, _, _ = load_embeddings_and_csv(query_path[3], "val", "audio", 768, 'frame')  #
    
    query_audio_sep_embeddings, train_sep_captions, train_sep_wav_paths, train_sep_tags = load_embeddings_and_csv(query_path[2], "train", "audio", 512, sep = True)
    query_val_sep_embeddings, val_sep_captions, val_sep_wav_paths, val_sep_tags = load_embeddings_and_csv(query_path[3], "val", "audio", 512, sep = True)
    
    # kb: audio512, text512, audio768
    kb_audio512_embeddings, kb_captions, kb_wav_paths = load_embeddings_and_csv(knowledge_base_path[0], "pretrain", "audio", 512)
    kb_text512_embeddings, _, _ = load_embeddings_and_csv(knowledge_base_path[0], "pretrain", "text", 512)
    # kb_audio768_embeddings, _, _ = load_embeddings_and_csv(knowledge_base_path[1], "pretrain", "audio", 768)
    # kb_frame768_embeddings, _, _ = load_embeddings_and_csv(knowledge_base_path[1], "pretrain", "audio", 768, 'frame') #
    
    # for seperating the train and val
    query_captions = {
                    # 'train': train_captions,
                    # 'val': val_captions,
                    # 'pretrain': kb_captions
                    'train_sep': train_sep_captions,
                    'val_sep': val_sep_captions
                    }
    query_wav_paths = {
                    # 'train': train_wav_paths,
                    # 'val': val_wav_paths,
                    # 'pretrain': kb_wav_paths
                    'train_sep': train_sep_wav_paths,
                    'val_sep': val_sep_wav_paths
                    }
    query_tags = {
        'train_sep': train_sep_tags,
        'val_sep': val_sep_tags  
    }

    # tensor
    print('convert datas to numpy array...')
    #query_audio_embeddings = torch.stack(query_audio_embeddings).numpy().astype('float32')
    query_audio_embeddings = query_audio_embeddings.astype('float32') # torch tensor로만 바꿔주기!
    # query_frame_embeddings = query_frame_embeddings.astype('float32')
    query_val_embeddings = query_val_embeddings.astype('float32')
    query_audio_sep_embeddings = query_audio_sep_embeddings.astype('float32')
    query_val_sep_embeddings = query_val_sep_embeddings.astype('float32')
    # query_val_frame_embeddings = query_val_frame_embeddings.astype('float32')
    
    
    kb_audio512_embeddings = kb_audio512_embeddings.astype('float32')
    kb_text512_embeddings = kb_text512_embeddings.astype('float32')
    # kb_audio768_embeddings = kb_audio768_embeddings.astype('float32')
    # kb_frame768_embeddings = kb_frame768_embeddings.astype('float32')
    
    # kb: wavcaps, audioset, clotho (train + pretrain)
    if config.index_args.big_kb:
        # big_kb_captions
        big_kb_captions = pd.concat([kb_captions, train_captions]).reset_index(drop = True)
        # big_kb_wav_paths
        big_kb_wav_paths = pd.concat([kb_wav_paths, train_wav_paths]).reset_index(drop = True)
        # big_kb_audio512_embeddings
        big_kb_audio512_embeddings = np.concatenate((kb_audio512_embeddings, query_audio_embeddings))
        # big_kb_text512_embeddings
        query_text_embeddings, _, _ = load_embeddings_and_csv(query_path[0], "train", "text", 512)
        big_kb_text512_embeddings = np.concatenate((kb_text512_embeddings, query_text_embeddings))
        # big_kb_audio768_embeddings
        # query_audio768_embeddings, _, _ = load_embeddings_and_csv(query_path[1], "train", "audio", 768)
        # big_kb_audio768_embeddings = np.concatenate((kb_audio768_embeddings, query_audio768_embeddings))
            
    
    # audio, text, frame
    index_types = config.index_args.index_types
    
    def make_index(embedding_dim, nlist):
        index_cpu = faiss.IndexFlatIP(embedding_dim)
        return index_cpu
    
    def create_indices(index_types, text_embedding_dim=config.index_args.text_dimension, audio_embedding_dim=config.index_args.audio_dimension, nlist=128):
        indices = {}

        for index_type in index_types:
            if index_type == "text":
                indices["text"] = make_index(text_embedding_dim, nlist)
            elif index_type == "audio":
                indices["audio"] = make_index(audio_embedding_dim, nlist)
            elif index_type == "frame":
                indices["frame"] = make_index(768, nlist)
            else:
                raise ValueError(f"Invalid index type: {index_type}")

        return indices
    
    # audio, text, frame index
    selected_indices = create_indices(index_types)
    
    # query: train
    # query_list = [query_audio_embeddings, query_val_embeddings]
    # query_frame_list = [query_frame_embeddings, query_val_frame_embeddings]
    # query_sep_list = [query_audio_sep_embeddings, query_val_sep_embeddings]
    # query: pretrain
    query_list = [query_val_embeddings]
    query_frame_list = []
    query_sep_list = [query_audio_sep_embeddings, query_val_sep_embeddings]
    
    # kb: pretrain + train
    kb_list = [big_kb_audio512_embeddings, big_kb_text512_embeddings] # big_kb_audio768_embeddings
    
    start_time = time.time()
    print('add vectors to index')
    # for kb_embeds, modality in zip(kb_list, index_types): # audio512, text512, audio768 index saved
    #     faiss.normalize_L2(kb_embeds)
    #     selected_indices[modality].add(kb_embeds)
    for kb_embeds, modality in zip(kb_list, index_types):
    # Convert to standard NumPy array
        kb_embeds_np = np.array(kb_embeds).copy().astype('float32')
        faiss.normalize_L2(kb_embeds_np)
        selected_indices[modality].add(kb_embeds_np)
    elapsed_time = time.time() - start_time
    print(f"elapsed time for add vectors to index: {elapsed_time}")
    
    # 
    index_list = ['train_sep', 'val_sep'] # 'train', 'val', 'pretrain', 'train_sep', 'val_sep'
    start_time = time.time()
    print('search')
    # [audio_train, audio_val]2audio, [audio_train, audio_val]2text, [frame_train, frame_val]2frame(audio768)
    for modality in tqdm(query_mode):
        split_string = modality.split("2")

        query_modality = split_string[0] # audio, frame(audio768)
        kb_modality = split_string[1] # audio, text, frame(audio768)
        for i, mode in enumerate(index_list):
            if query_modality == 'frame' and kb_modality == 'frame': # B, 4, 768
                query_frame_embeddings = query_frame_list[i]
                query_frame_embeddings_np = np.array(query_frame_embeddings).copy().astype('float32')
                num_frames = query_frame_embeddings_np.shape[1] # 4
                B = query_frame_embeddings_np.shape[0] # B
                top_k_per_frame = 7 #
                final_top_k = 6
                all_indices = []

                for frame_idx in range(num_frames):
                    frame_embeddings = query_frame_embeddings_np[:, frame_idx, :]
                    frame_embeddings = np.ascontiguousarray(frame_embeddings)
                    faiss.normalize_L2(frame_embeddings)
                    _, I = selected_indices[kb_modality].search(frame_embeddings, top_k_per_frame)
                    all_indices.append(I)
                    
                def combine_indices_ordered(all_indices, top_k_per_frame, final_top_k):
                    B = all_indices[0].shape[0]  # Number of batches
                    num_frames = len(all_indices)
                    combined_indices = np.zeros((B, final_top_k), dtype=np.int64) - 1  # Initialize with -1

                    for b in range(B): # batch
                        unique_indices = []
                        for k in range(top_k_per_frame): # frame
                            for frame_idx in range(num_frames):
                                index = all_indices[frame_idx][b, k] # top1
                                if index not in unique_indices:
                                    unique_indices.append(index)
                                    if len(unique_indices) == final_top_k:
                                        break
                            if len(unique_indices) == final_top_k:
                                break
                        combined_indices[b, :len(unique_indices)] = unique_indices

                    return combined_indices
                I = combine_indices_ordered(all_indices, top_k_per_frame, final_top_k)           
            elif mode == 'train_sep' or mode == 'val_sep':
                query_audio_sep_embeddings = query_sep_list[i]
                query_audio_sep_embeddings_np = np.array(query_audio_sep_embeddings).copy().astype('float32')
                max_tag_size = query_audio_sep_embeddings_np.shape[1] # 4
                B = query_audio_sep_embeddings_np.shape[0] # B
                top_k_per_tag = 7 #
                final_top_k = 6
                all_indices = []
                all_distances = []

                for i in tqdm(range(B)):
                    non_zero_indices = np.any(query_audio_sep_embeddings_np[i] != 0, axis=1)
                    sep_embeddings = query_audio_sep_embeddings_np[i,non_zero_indices, :]  # Extract non-padded events
                    sep_embeddings = np.ascontiguousarray(sep_embeddings)
                    faiss.normalize_L2(sep_embeddings)
                    D, I = selected_indices[kb_modality].search(sep_embeddings, top_k_per_tag)
                    # D는 각 tag 별로 top_k_per_tag개의 결과를 가지게 된다.
                    # 모든 tag의 결과로 하나의 audio에 대한 검색 결과를 만든다.
                    # 각 tag 별로 distance를 0.79 혹은 평균값과 비교하여 남긴다.
                    # tag의 distance list의 평균값과 0.79를 비교해서 0.79가 낮으면 0.79와 비교, 반대로 0.79가 높으면 평균값과 비교한다.
                    # 
                    # 이러한 distance를 기준으로 I를 조정한다.
                    # 예를 들어서, 아래처럼 D와 I가 나왔다고 하자.
                    # D
                    # array([[0.9402561 , 0.93553644, 0.9227829 , 0.9224911 , 0.917307  ,
                    #         0.8590025 , 0.7881237 ],
                    #     [0.8090186 , 0.79449075, 0.7936646 , 0.7932924 , 0.7930682 ,
                    #         0.7924906 , 0.7916684 ],
                    #     [0.79242194, 0.76680744, 0.743369  , 0.7427633 , 0.74210036,
                    #         0.741607  , 0.741607  ]], dtype=float32)
                    # I
                    # array([[340214, 340216, 340215, 340217, 340213, 123636, 347067],
                    #     [224298, 300076, 300152, 306865, 327206, 329126, 312165],
                    #     [  7469, 312651, 256887, 244176, 251859, 304886, 247985]])
                    
                    # 첫번째 태그에서 D[0].mean()가 0.89792854이므로 0.79보다 커서 0.79와 비교하면 [340214, 340216, 340215, 340217, 340213, 123636]이 남는다
                    # 두번째 태그에서 D[1].mean()가 0.7953848이므로 0.79보다 커서 0.79와 비교하면 [224298, 300076, 300152, 306865, 327206, 329126, 312165]이 남는다.
                    # 세번째 태그에서 D[2].mean()이 0.7529538이므로 0.79보다 작아서 세번째 태그의 모든 값을 평균값과 비교해보면 [  7469, 312651]만 남는다.
                    # 여기서 top1 -> top2 -> top3 ... 으로 정리가 된다. 최대 6개를 남긴다.
                    # 340214, 224298, 7469, 340216, 300076, 312651 가 최종 리스트가 된다.
                    # 이를 all_indices에 추가한다. all_indices.append()
                    # 최종적으로는 B만큼의 ranking list완성된다.
                    # Filter indices based on threshold or mean distance
                    threshold = 0.79
                    filtered_indices = []
                    temp_indices = []
                    for distances, indices in zip(D, I):
                        mean_distance = distances.mean()
                        # 각 태그를 하기 위해서 점수가 낮은 tag도 위로 올리는 방식 Noisy가 껴서 성능에는 좋지 않을 수 있다.
                        valid_indices = indices[(distances >= threshold) | (distances >= mean_distance)]
                        # 그냥 relevance가 무조건 높아야하는 방식
                        # valid_indices = indices[(distances >= threshold)]
                        filtered_indices.append(valid_indices)  # Keep up to final_top_k items
                        if mean_distance < threshold:
                            low = mean_distance - 1
                            temp_valid_indices = indices[(distances >= low) & (distances <= threshold)]
                            temp_indices.extend(temp_valid_indices)
                    
                    # Interleave the top results from each tag, avoiding duplicates
                    interleaved_indices = []  # Using a set to avoid duplicates
                    for idx in range(final_top_k):
                        for tag_indices in filtered_indices:
                            if idx < len(tag_indices):
                                if tag_indices[idx] not in interleaved_indices:  # Check for duplicates
                                    interleaved_indices.append(tag_indices[idx])
                                    if len(interleaved_indices) >= final_top_k:
                                        break
                        if len(interleaved_indices) >= final_top_k:
                            break
                    if len(interleaved_indices) < final_top_k:
                        interleaved_indices.extend(temp_indices)
                    all_indices.append(interleaved_indices[:final_top_k])
                I = np.array(all_indices)

                # Check if the shape is as expected, if not, pad with the first ranking element
                # for i in range(all_indices_np.shape[0]):
                #     if len(all_indices_np[i]) < final_top_k:
                #         first_ranking_element = all_indices_np[i][0]
                #         padding = np.full((final_top_k - len(all_indices_np[i])), first_ranking_element)
                #         all_indices_np[i] = np.concatenate((all_indices_np[i], padding))

                # Ensure the shape is (B, final_top_k)
                # I = np.array(all_indices_np.tolist())
            else: # audio2audio, text2text # B, 512
                query_audio_embeddings = query_list[i]
                query_audio_embeddings_np = np.array(query_audio_embeddings).copy().astype('float32')
                faiss.nordmalize_L2(query_audio_embeddings_np) # B, 512
                D, I = selected_indices[kb_modality].search(query_audio_embeddings_np, config.index_args.top_k)
            create_json_file_retrieval(config, I, query_modality, kb_modality, query_captions[mode], query_wav_paths[mode], big_kb_captions, big_kb_wav_paths, mode)
    elapsed_time = time.time() - start_time
    print(f"elapsed time for search: {elapsed_time}")
    save_index(selected_indices, config)
    return retrieved_result

def generate_faiss_index(config, dataloader, device, mode):
    """
    Generate faiss index for a PyTorch DataLoader.

    Parameters:
    - dataloader: PyTorch DataLoader producing embeddings
    - embedding_dim: 512차원으로 통일했습니다. / audio만 768차원이 사용 가능하다. 
    - pretrain.yaml 파일
    index_args:
    index_save_path: "./data/index"
    index_types: ["audio"]
    audio_dimension: 768 # 768 or 512
    text_dimension: 512 # 512

    Returns:
    - captions N개, wav_paths N개, selected_indices는 index_types에서 선택한 index를 생성할 수 있습니다.
    - AudioRAG 폴더에 caption_wav_path.csv(captions, wav_paths)가 저장되고, AudioRAG/data/index 폴더에 audio_faiss_index.bin, text_faiss_index.bin이 저장됩니다.
    """
    #
    # FAISS index
    index_types = config.index_args.index_types
    #
    # function
    def make_index(embedding_dim, nlist):
        # quantizer = faiss.IndexFlatL2(embedding_dim)
        # index_cpu = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        index_cpu = faiss.IndexFlatIP(embedding_dim)
        return index_cpu
    #
    # nlist: 32~512, trade-off between search time, nprobe=32~128
    def create_indices(index_types, text_embedding_dim=config.index_args.text_dimension, audio_embedding_dim=config.index_args.audio_dimension, nlist=128):
        indices = {}
        if "pair" in index_types:
            pair_embedding_dim = text_embedding_dim + audio_embedding_dim

        for index_type in index_types:
            if index_type == "text":
                indices["text"] = make_index(text_embedding_dim, nlist)
            elif index_type == "audio":
                indices["audio"] = make_index(audio_embedding_dim, nlist)
            elif index_type == "pair":
                indices["pair"] = make_index(pair_embedding_dim, nlist)
            else:
                raise ValueError(f"Invalid index type: {index_type}")

        return indices
    #
    selected_indices = create_indices(index_types)
    captions = []
    wav_paths = []

    # model
    clap = CLAP_Module(enable_fusion=True, device = device)  # 615M
    clap.load_ckpt()  # download the default pretrained checkpoint.

    modalities = {
        "text": {"process": process_text_data, "embeddings": [], "data_key": "caption"}, # text embeddings
        "audio": {"process": process_audio_data, "frame_process": process_frame_data, "embeddings": [], "frame_embeddings": [], "data_key": "audio_sample"} # audio embeddings
        # 새로운 모달을 추가할 수 있다.
    }

    # index types에 있는 종류를 embedding으로 바꾼다.
    # # for test with samples
    # from itertools import islice
    # num_batches_to_test = 50
    # tqdm(islice(dataloader, num_batches_to_test)):
    # tqdm(dataloader):
    
    name_map = {
        'audio_sample': 0,
        'caption': 1,
        'wav_path': 2
    }

    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(dataloader):# batch -> audio_sample, caption, wav_path 
            captions.extend(batch[name_map["caption"]])
            wav_paths.append(batch[name_map["wav_path"]])
            for modality, info in modalities.items():
                if modality in index_types:
                    data = batch[name_map[info["data_key"]]]
                    if modality == 'audio':
                        data = [item['waveform'] for item in data]
                    outputs = info["process"](data, clap, config) # audio
                    # frame 코드
                    if modality == 'audio' and config.index_args.query_mode == 'frame2audio':
                        frame_outputs = info["frame_process"](data, clap, config) 
                        info["frame_embeddings"].extend(frame_outputs.cpu().contiguous())
                    info["embeddings"].extend(outputs.cpu().contiguous())

    wav_paths_list = [item for sublist in wav_paths for item in sublist]
    
    
    ## 여기까지 저장.
    # 인덱스 만드는 코드가 여기 있어야겠네.
    # faiss indices에 저장한다.
    for modality in index_types: # audio, text
        print('add vectors to index')
        embeddings = torch.stack(modalities[modality]["embeddings"]).numpy().astype('float32') 
        # selected_indices[modality].train(embeddings)
        faiss.normalize_L2(embeddings)
        selected_indices[modality].add(embeddings)
        elapsed_time = time.time() - start_time
        print(f"elapsed time for {modality}_faiss_index: {elapsed_time}")
        print('search')
        ###
        if modality == 'audio' and config.index_args.query_mode == 'frame2audio': # B, 4, 768
            frame_embedding = torch.stack(modalities[modality]["frame_embeddings"]).numpy().astype('float32') 
            num_frames = frame_embedding.shape[1] # 4
            B = frame_embedding.shape[0] # B
            top_k_per_frame = 7 #
            final_top_k = 6
            all_indices = []

            for frame_idx in range(num_frames):
                frame_embeddings = frame_embedding[:, frame_idx, :]
                frame_embeddings = np.ascontiguousarray(frame_embeddings)
                _, I = selected_indices[modality].search(frame_embeddings, top_k_per_frame)
                all_indices.append(I)
                
            def combine_indices_ordered(all_indices, top_k_per_frame, final_top_k):
                B = all_indices[0].shape[0]  # Number of batches
                num_frames = len(all_indices)
                combined_indices = np.zeros((B, final_top_k), dtype=np.int64) - 1  # Initialize with -1

                for b in range(B): # batch
                    unique_indices = []
                    for k in range(top_k_per_frame): # frame
                        for frame_idx in range(num_frames):
                            index = all_indices[frame_idx][b, k] # top1
                            if index not in unique_indices:
                                unique_indices.append(index)
                                if len(unique_indices) == final_top_k:
                                    break
                        if len(unique_indices) == final_top_k:
                            break
                    combined_indices[b, :len(unique_indices)] = unique_indices

                return combined_indices
            I = combine_indices_ordered(all_indices, top_k_per_frame, final_top_k)           
        else:  # Standard case
            D, I = selected_indices[modality].search(embeddings, config.index_args.top_k)
        create_json_file_retrieval(config, modality, I, wav_paths_list, captions, mode)
    
    return selected_indices, captions, wav_paths_list


# def process_item(args):
#     query_index, topk_indices, query_wav_paths, query_captions, kb_wav_paths, kb_captions = args
#     query_audio_path = query_wav_paths[query_index]
#     query_caption = query_captions[query_index]

#     topk_items = []
#     for i in topk_indices:
#         kb_item = (kb_wav_paths[i], kb_captions[i])
#         if kb_item != (query_audio_path, query_caption):
#             topk_items.append(kb_item)
#         if len(topk_items) == 5:
#             break

#     return query_audio_path, topk_items

# def create_json_file_retrieval(config, I, query_modality, kb_modality, query_captions, query_wav_paths, kb_captions, kb_wav_paths): # config, modality, I, wav_paths_list, captions
    
#     results = {}
    
#     print('json files save...')
#     start_time = time.time()

#     # my code
#     # for query_index, topk_indices in tqdm(enumerate(I)):
#     #     query_audio_path = query_wav_paths[query_index]
#     #     query_caption = query_captions[query_index]
        
#     #     query = (query_audio_path, query_caption)

#     #     topk_items = []
#     #     for i in topk_indices:
#     #         kb_item = (kb_wav_paths[i], kb_captions[i])
#     #         if kb_item != query:
#     #             topk_items.append(kb_item)
#     #         if len(topk_items) == 5:
#     #             break

#     #     results[query_audio_path] = topk_items

            
#     # multiprocessing code
#     # Prepare the arguments for each process
#     args = [(query_index, topk_indices, query_wav_paths, query_captions, kb_wav_paths, kb_captions) for query_index, topk_indices in enumerate(I)]

#     # Determine the number of processes based on the available CPU cores
#     num_processes = min(cpu_count(), len(I))

#     # Use multiprocessing to process the data in parallel
#     with Pool(num_processes) as pool:
#         for query_audio_path, topk_items in tqdm(pool.imap_unordered(process_item, args), total=len(args)):
#             results[query_audio_path] = topk_items    
    
#     elapsed_time = time.time() - start_time
#     print(f"elapsed time for saving json files_{query_modality}2{kb_modality}: {elapsed_time}")
#     print('json files save complete...')
#     check_and_create_directory(config.index_args.index_save_path)
        
#     with open(f'{config.index_args.index_save_path}/{query_modality}2{kb_modality}.json', 'w') as file:
#         json.dump(results, file, indent =4)
    
# Define process_item at the top level
def process_item(args):
    query_index, topk_indices, query_wav_paths, query_captions, kb_wav_paths, kb_captions = args
    query_audio_path = query_wav_paths[query_index]
    query_caption = query_captions[query_index]

    topk_items = []
    for i in topk_indices:
        kb_item = (kb_wav_paths[i], kb_captions[i])
        if kb_item != (query_audio_path, query_caption):
            topk_items.append(kb_item)
        if len(topk_items) == 5:
            break

    return query_audio_path, topk_items

# Your create_json_file_retrieval function
def create_json_file_retrieval(config, I, query_modality, kb_modality, query_captions, query_wav_paths, kb_captions, kb_wav_paths, mode): 
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
            if len(topk_items) == 5:
                break

        results[query_audio_path] = topk_items  

    elapsed_time = time.time() - start_time
    print(f"elapsed time for saving json files_{query_modality}2{kb_modality}_{mode}: {elapsed_time}")
    print('json files save complete...')

    # Assuming check_and_create_directory is a function you have defined elsewhere
    check_and_create_directory(config.index_args.index_save_path)
        
    with open(f'{config.index_args.index_save_path}/{query_modality}2{kb_modality}_{mode}.json', 'w') as file:
        json.dump(results, file, indent=4)
        
        
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


if __name__ == "__main__":
    config = get_config()
    print(config)
    
    # query (train, validation)
    # train: audio512 embedding, audio_frame embedding O
    # validation: audio512 embedding, audio_frame embedding X
    
    # kb (pretrain)
    # pretrain: audio512 embedding, audio768 embedding, text512 embedding O
    
    # dataloader
    print('data loading...')
    dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0, seperation = config.index_args.seperation)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    # # # save embeddings
    modalities, captions, wav_paths_list = precompute_and_save(config, dataloader, device, "pretrain")
    
    # # retrieve
    # 아래 kb는 pretrain 두개밖에 없지만 실제로는 query_path에서 가져온 것도 kb로 사용함.
    # knowledge_base_path = ["./data/pretrain_512_embeddings", "./data/pretrain_768_embeddings"]
    # query_path = ["./data/train_512_embeddings", "./data/train_768_embeddings", "./data/train_512_embeddings_sep_max_7", "./data/val_512_embeddings_sep_audiocaps", "./data/val_512_embeddings"] # + validation set #"./data/val_512_embeddings_macs", "./data/val_768_embeddings_macs"]
    # make_retrieval_result_jsonfiles(config, query_path, knowledge_base_path)
    
    # pass 
    # selected_indices, captions_list, wav_paths = generate_faiss_index(config, dataloader, device, mode = 'pretrain')
    # save_index(selected_indices, captions_list, wav_paths, config, mode = 'pretrain') # pretrain or train
    
    # #
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()
    # index = RetrievalIndex(n_probe=128, index_path="./data/pretrain_index", audio_index_file = "audio_512_faiss_index.bin", text_index_file = "text_512_faiss_index.bin", index2_file = "caption_wav_pretrain_512_path.csv", top_k=5, query_mode="audio2audio", device = device)
    # index.make_query_topk_npy(config)
    # test
    # text_data = ["a dog is barking at a man walking by", "Wind and a man speaking are heard, accompanied by buzzing and ticking.", 'water is falling from a roof and light wind can be heard in the background ']
    # audio_file = ["./examples/yapping-dog.wav", "./examples/Yb0RFKhbpFJA.flac", '/drl_nas1/ckddls1321/data/WavCaps/waveforms/FreeSound_flac/486953.flac']

    # def get_audio_embedding_before_projection(self, data):
    #         """Get the audio embedding from the model

    #         Parameters
    #         ----------
    #         data: a list of dict
    #             the audio input dict list from 'get_audio_feature' method

    #         Returns
    #         ----------
    #         audio_embed: torch.Tensor
    #             a tensor of audio_embeds (N, D)

    #         """
    #         device = next(self.parameters()).device
    #         input_dict = {}
    #         keys = data[0].keys()
    #         for k in keys:
    #             input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
    #         audio_embeds = self.encode_audio(input_dict, device=device)["embedding"] # projection 전을 찾기 위함이다.
    #         return audio_embeds
    # def get_audio_embedding_frame(self, data):
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
    #             device = next(self.parameters()).device # self.device?
    #             input_dict = {}
    #             keys = data[0].keys()
    #             for k in keys:
    #                 input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
    #             audio_embeds = self.encode_audio(input_dict, device=device)["fine_grained_embedding"]
    #             return audio_embeds
    #         # self.clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_frame, self.clap.model)
    # clap_model = CLAP_Module(enable_fusion=True)  # 615M
    # clap_model.load_ckpt()
    # # 768
    # # clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model) 
    # # frame 768
    # # clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_frame, clap_model.model) 
    
    # clap_model.eval()
    # with torch.no_grad():
    #     # text
    #     # text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
    #     # print(text_embed)
    #     # print(text_embed.shape)

    #     # audio       
    #     audio_embed = clap_model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
        
    #     # frame
    #     # chunks = audio_embed.chunk(4, dim=1)
    #     # averaged_chunks = [chunk.mean(dim=1, keepdim=True) for chunk in chunks]
    #     # audio_embed = torch.cat(averaged_chunks, dim=1).permute(1, 0, 2)
    #     print(audio_embed)
    #     print(audio_embed.shape)

    # audio_index_path = "train_512_index" # pretrain_768_index
    # audio_file_name = "audio_512_faiss_index.bin" # audio_768_faiss_index
    # text_index_path = "train_512_index"
    # text_file_name = "text_512_faiss_index.bin"
    # csv_file_path = "caption_wav_train_512_path.csv"
    # n_probe = 128

    # # audio_512_index = faiss.read_index(f"./data/{audio_index_path}/{audio_file_name}")
    # audio_512_index = faiss.read_index(f"./data/{audio_index_path}/{audio_file_name}")
    # # text_512_index = faiss.read_index(f"./data/{text_index_path}/{text_file_name}")
    # audio_512_index.nprobe = n_probe
    # # text_512_index.nprobe = n_probe
    # # audio_768_index.nprobe = n_probe

    # def load_caption_wav_mapping(csv_path):
    #     df = pd.read_csv(csv_path)
    #     return df['caption'].tolist(), df['wav_path'].tolist()

    # def check_nearest_neighbors(index, queries, k, captions, wav_paths):
    #     # Search the index
    #     D, I = index.search(queries, k)
    #     for i, neighbors in enumerate(I):
    #         print(f"Query {i}:")
    #         for neighbor in neighbors:
    #             print(f" - Neighbor id: {neighbor}, Caption: {captions[neighbor]}, Wav path: {wav_paths[neighbor]}")
    #         print(f" - Distances: {D[i]}")

    # def frame_check_nearest_neighbors(index, queries, k, captions, wav_paths):
    #     #frame
    #     import numpy as np
    #     # Search the index
    #     for _, queries_embed in enumerate(queries):
    #         queries_embed = np.ascontiguousarray(queries_embed)
    #         D, I = index.search(queries_embed, k)
        
    #         for i, neighbors in enumerate(I):
    #             print(f"Query {i}:")
    #             for neighbor in neighbors:
    #                 print(f" - Neighbor id: {neighbor}, Caption: {captions[neighbor]}, Wav path: {wav_paths[neighbor]}")
    #             print(f" - Distances: {D[i]}")

    # captions_list, wav_paths_list = load_caption_wav_mapping(f"./data/{audio_index_path}/{csv_file_path}")

    # # text_query_embeddings = text_embed.cpu().detach().numpy().astype('float32')
    # audio_query_embeddings = audio_embed.cpu().detach().numpy().astype('float32')
    # # text_embed

    # k = 5
    # # # # text2text
    # # # check_nearest_neighbors(text_index, text_query_embeddings, k, captions_list, wav_paths_list)
    # # # # text2audio
    # # # check_nearest_neighbors(audio_index, text_query_embeddings, k, captions_list, wav_paths_list)
    # # # audio2audio
    # check_nearest_neighbors(audio_512_index, audio_query_embeddings, k, captions_list, wav_paths_list)
    # audio2text
    # check_nearest_neighbors(text_512_index, audio_query_embeddings, k, captions_list, wav_paths_list)
    # frame2audio
    # frame_check_nearest_neighbors(audio_768_index, audio_query_embeddings, k, captions_list, wav_paths_list)
    
    
    # # test2 for retrievalIndex class
    # index = RetrievalIndex()
    
    # audio_query_embedding = index.query_embedding(audio_embed)
    # text_query_embedding = index.query_embedding(text_embed)
    
    # index.get_nns("audio", audio_query_embedding, k = 16, show = True)
    # index.get_nns("text", audio_query_embedding, k = 16, show = True)