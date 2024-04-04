import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from utils.utils import fix_seed, get_config, AverageMeter
from dataset.dataload import load_dataset, load_dataloader
from models.model import CLAP2LLAMA

import argparse
from tqdm import tqdm
import sys
import time
from omegaconf import OmegaConf
from datetime import datetime

import torch
from torch import optim
from accelerate import Accelerator
from accelerate.utils import gather_object
import transformers
import evaluate
# 수정이 필요하다.
from metrics import SpiceMetric, CocoTokenizer, CiderMetric
# accelerator
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=False)
# # accelerator에 gradient accumulation이랑 wandb 연동이랑 ddp_kwargs도 사용하고 even batches까지? 이런 건 어떻게 한 걸까? 내가 일단 accelerator을 사용하는데 거기에 세팅을 추가한 건가
# accelerator = Accelerator(gradient_accumulation_steps=8, log_with="wandb", kwargs_handlers=[ddp_kwargs], even_batches=True)
accelerator = Accelerator(log_with="wandb")


@torch.no_grad
def validate(val_dataloader, model):
    # 다른 여러 github으로 evaluation해보자.
    model.eval() # x
    sacrebleu = evaluate.load('sacrebleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load("rouge")
    spice = SpiceMetric()
    cider = CiderMetric()
    
    # 이거 꼭 해야하나? distributed setting에서 벗어나기 위해서 사용한다. 모델 저장 전에 사용한다.
    unwrapped_model = accelerator.unwrap_model(model)
    
    gen_captions = []
    ref_captions = []
    
    # 데이터 샘플링 (나중에 retrieved result도 확인하기)
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)): # len(val_dataloader)
        audio, audio_path, caption = batch
        with accelerator.autocast():
            # 모델에 넣고 generate
            generated_caption = unwrapped_model.generate_caption_inference(audio)
            # if epoch > 7:
            print(generated_caption)
            print(caption)
            # generate cpations에 저장
            gen_captions.extend(generated_caption)
            ref_captions.extend(caption)
    
    if accelerator.is_main_process:
        sacrebleu_score = sacrebleu.compute(predictions=gen_captions, references=ref_captions)
        meteor_score = meteor.compute(predictions=gen_captions, references=ref_captions)
        rouge_score = rouge.compute(predictions=gen_captions, references=ref_captions)
        # ?
        tokenizer = CocoTokenizer(gen_captions, ref_captions)
        tokens = tokenizer.tokenize()
        if isinstance(ref_captions, list) and all(isinstance(caption, str) for caption in ref_captions):
            ref_captions = [[caption] for caption in ref_captions] # List of list, val or test may include 5 captions
        spice_score = spice.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        cider_score = cider.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])
        # ?
        metrics_all = {
            "sacrebleu": sacrebleu_score['score'], # 좀 비정상적인듯? 아닌감..
            "meteor": meteor_score['meteor'],
            "spice": spice_score['average_score'],
            "cider": cider_score['score'],
            "rougeL": rouge_score['rougeL'],
            "spider": spider_score,
        }
        accelerator.print(metrics_all)
        accelerator.log(metrics_all)
        return metrics_all
    else:
        return None

def train(epoch, train_dataloader, model, optimizer, lr_scheduler, clip_norm):
    model.train()
    epoch_loss = AverageMeter()
    epoch_start_time = time.time()
    
    # from itertools import islice
    # num_iteration = 10
    # train_dataloader = islice(train_dataloader, num_iteration)
    ####
    pbar = tqdm(enumerate(train_dataloader), desc = "iteration training", total = int(len(train_dataloader))) # int(len(train_dataloader))
    for idx, batch in pbar:
        # step마다 lr이 조정되는 값을 확인하기 위해서 사용한다.
        step = len(train_dataloader) * (epoch - 1) + idx # len(train_dataloader)
        iter_start_time = time.time()
        # gradient accumulation
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            audio, audio_path, caption = batch
            # mixed precision
            with accelerator.autocast(): # 이건 지워도 보고 그냥도 해보자. 성능 차이를 확인해보자.
                outputs = model(audio, caption)
            loss = outputs['loss']
            accelerator.backward(loss)
            # gradient cliping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), clip_norm)
                accelerator.log({"lr": optimizer.param_groups[0]["lr"]}, step=step) # step마다 cliping을 통해 lr이 조정되나? 
            optimizer.step()
            lr_scheduler.step()
        iter_end_time = iter_start_time - time.time()
        # tqdm bar
        pbar.set_description(f"loss: {epoch_loss.avg}, time: {iter_end_time}")
        epoch_loss.update(loss.cpu().item())
    # wadb tracking
    accelerator.log({"loss": epoch_loss.avg, "epoch": epoch})
    
    epoch_end_time = epoch_start_time - time.time()
    print(f"elapsed time for one epoch: {epoch_end_time}")

@torch.no_grad
def test(train_dataloader, model):
    model.eval()
    pbar = tqdm(enumerate(train_dataloader), desc = 'test', total = int(len(train_dataloader)))
    for idx, batch in pbar:
        audio, path, caption = batch
        with accelerator.autocast():
            outputs = model(audio, caption)
    
def main():
    # parser
    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    if accelerator.is_main_process:
        print(config)
    
    # seed 설정
    fix_seed(config.seed)
    
    # wandb
    today_str = datetime.now().strftime("%Y%m%d")
    exp_name = 'pretrain' + "_" + config.model_args.align.model_name + f"_lr_{config.optim_args.lr}_batch_{config.data_args.global_batch_size}_seed_{config.seed}_{today_str}"
    accelerator.init_trackers(
        project_name="audio-captioning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": {"name": exp_name}}
    )
    
    # 데이터셋 불러오기
    train_dataset = load_dataset(config.pretrain_dataset, config.filtering, config.train)
    val_dataset = load_dataset(config.val_dataset, config.filtering)
    
    # 데이터로더 불러오기
    train_dataloader = load_dataloader(config, train_dataset, 'train')
    val_dataloader = load_dataloader(config, val_dataset, 'val')
    
    # 모델 불러오기
    model = CLAP2LLAMA(config.model_args)
    # model.to(device)

    # trained weights (align module을 특별하게 만들기 위해서 따로 뺴놓은듯)
    if config.model_args.checkpoint_path:
        model.load_ckpt(config.model_args.checkpoint_path)
    
    # 학습 준비
    # accumulation steps, train steps, warm up steps
    accelerator.gradient_accumulation_steps = config.data_args.global_batch_size // (config.data_args.batch_size * accelerator.state.num_processes)
    warmup_steps = int(len(train_dataloader)) * config.training.warmup_epochs // accelerator.gradient_accumulation_steps
    train_steps = int(len(train_dataloader)) * config.training.epochs // accelerator.gradient_accumulation_steps
    # optimizer, lr_scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay, fused=False)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    # acculerator 사용
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, scheduler) # model train_dataloader # 이걸 실행하면 메모리 에러가 난다. 그러니깐 accelerator가 어떤 문제를 발생시키는데?
    # load clap model pretrained weight -> 애초에 모델 불러오는 부분이 문제임.
    # model.encoder.clap.load_ckpt()
    
    # test
    # test(train_dataloader, model)
    
    # eval only code
    if config.training.eval:
        validate(val_dataloader, model)
        accelerator.wait_for_everyone()
        accelerator.end_training()
        sys.exit()
    
    # 학습하기 (나중에 wandb)
    spiders = []
    for epoch in tqdm(range(1, config.training.epochs + 1), desc = "epoch training"):
        train(epoch, train_dataloader, model, optimizer, scheduler, clip_norm = config.training.clip_grad)
        if config.training.validate:
            # from itertools import islice
            # sliced_val_dataloader = islice(val_dataloader, 10)
            metrics_all = validate(val_dataloader, model)
            if accelerator.is_main_process:
                spiders.append(metrics_all["spider"])
                save_ckpt = metrics_all["spider"] >= max(spiders) 
                if save_ckpt:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_ckpt(config.training.output_path)
        accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
    
    