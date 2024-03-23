from utils.utils import fix_seed, get_config
from dataset.dataload import load_dataset, load_dataloader
from models.model import CLAP2LLAMA
import argparse
from tqdm import tqdm
import torch

device = torch.device('cuda:1')

def main():
    # parser
    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    print(config)
    
    # seed 설정
    fix_seed(config.seed)
    
    # 데이터셋 불러오기
    train_dataset = load_dataset(config.train_dataset, config.filtering, config.train)
    
    # 데이터로더 불러오기
    train_dataloader = load_dataloader(config, train_dataset)
    
    # 모델 불러오기
    model = CLAP2LLAMA(config.model_args)
    model.to("cuda:0")

    # trained weights (align module을 특별하게 만들기 위해서 따로 뺴놓은듯)
    if config.model_args.checkpoint_path:
        model.load_ckpt(config.model_args.checkpoint_path)
    
    
    # 학습 준비
    
    # 학습하기 
    # wandb 사용, 
    for idx, batch in tqdm(enumerate(train_dataloader)):
        audio, audio_path, caption = batch
        # audio_path는 일단 필요없음
        output = model(audio, caption)
        
    
    # 결과 저장하기
    # config 내용도 함께 저장하면 좋겠다.
    
    # validation하기

if __name__ == "__main__":
    main()
    
    
    
