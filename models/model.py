import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, IA3Config, TaskType, PeftModel
from peft import get_peft_model

import librosa
import argparse
from omegaconf import OmegaConf
import os

from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig

##
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        # self.layer_norm = nn.LayerNorm(d_out)
        # self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        # init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.linear2(F.gelu(embed1))
        #embed2 = self.drop(self.linear2(F.gelu(embed1)))
        #embeds = self.layer_norm(embed1 + embed2)
        return embed2
    
# save, load도 따로 만들어줘야하나?
class CLAP2LLAMA(nn.Module):
    def __init__(self, config):
        super(CLAP2LLAMA, self).__init__()
        self.config = config
        self.device = config.device
        
        # 레이어 선언
        # audio encoder
        self.encoder_config = CLAPEncoderConfig.from_dict(OmegaConf.to_container(config.encoder, resolve=True))
        self.encoder = CLAPAudioTower(self.encoder_config)
        
        # decoder
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", add_eos_token = True, add_bos_token = True)
        if self.tokenizer.pad_token == '<unk>':
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.decoder.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = self.config.decoder.sequence_max_length
            #self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.task_prompt = self.config.task_prompt
        
        # alignment module: mlp
        if self.config.align.model_name == "MLP":
            # pretrain을 다시 하면...
            # Projection()
            
            # train을 하면... # encoder_config 설정해주기!
            modules = [
                nn.Linear(self.encoder_config.hidden_size, self.decoder.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
            ]
            self.enc_to_dec_proj = nn.Sequential(*modules)
            self.forward_align = self.forward_mlp
        
        # retrieved context
        # pass #
        
        # freeze parameters (audio, align, language)
        self.freeze_am = config.freeze_am
        self.freeze_lm = config.freeze_lm
        self.unfreeze_am = config.unfreeze_am
        
        # 1. freeze clap
        if self.freeze_am:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False
                if any(name.startswith(prefix) for prefix in config.unfreeze_am):
                    p.requires_grad = True
                    
        # 2. freeze align
        # for name, p in self.enc_to_dec_proj.named_parameters():
        #     p.requires_grad = False
        
        # 3. freeze language (or lora tuning, 알아서 freeze)
        # if self.freeze_lm:
        #     self.decoder.eval()
        #     for p in self.decoder.parameters():
        #         p.requires_grad = False
        # else:
        #     peft_type = config.peft_config.peft_type
        #     if peft_type == "LORA":
        #         peft_config = LoraConfig(**config.peft_config) # task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        #         # 아래 두 개는 왜하지? 이게 다른 논문들에서는 어떻게 lora tuning 했는지를 알아야겠다.
        #         self.decoder.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
        #         self.decoder.base_model.model.lm_head.original_module.weight.requires_grad = False
        #     if peft_type == "IA3":
        #         self.peft_config = IA3Config(**config.peft_config)
        #     self.decoder = get_peft_model(self.decoder, peft_config)    
        #     self.decoder.print_trainable_parameters()
                
        # memory 
        # checkpointing
        # self.decoder.gradient_checkpointing_enable()
        
    def load_ckpt(self, checkpoint_path):
        self.enc_to_dec_proj.load_state_dict(torch.load(checkpoint_path + "enc_to_dec_proj.bin", map_location=self.device), strict = True)
        if self.unfreeze_am:
            file_path = os.path.join(checkpoint_path, "audio_encoder.bin")
            if os.path.exists(file_path):
                self.encoder.load_state_dict(torch.load(file_path), strict=False)       
        # 이것도 해야하나? 일단 pass 
        # if not self.freeze_lm and 'finetuned' in checkpoint_path:
        #     print("Load LORA model")
        #     self.decoder = PeftModel.from_pretrained(self.decoder.base_model, checkpoint_path, config=self.peft_config)  # suppose don't use get_peft_model        
        
    def forward_encoder(self, audio):
        outputs = self.encoder(audio).last_hidden_state
        return outputs
    
    def forward_mlp(self, x):
        outputs = self.enc_to_dec_proj(x)
        return outputs
    
    def shift_tokenized_input(self, input_ids, attn_mask, audio_token_length, device):
        # 개인적으로 audio token이 너무 작은듯. 이 갯수를 더 늘려주는 게 좋을 것 같은데 다른데도 이렇게 1 토큰으로 집어넣나?
        batch_size, sequence_length = input_ids.shape[0], input_ids.shape[1]
        shifted_input_ids = torch.zeros((batch_size, sequence_length + audio_token_length), dtype = int).to(device)
        shifted_input_ids[:, audio_token_length:] = input_ids.clone()
        shifted_input_ids[:, :audio_token_length] = -100
        pad_mask = (shifted_input_ids == self.tokenizer.pad_token_id)
        shifted_input_ids.masked_fill_(pad_mask, -100)
        shifted_attn_mask = torch.ones((batch_size, sequence_length + audio_token_length), dtype = int).to(device)
        shifted_attn_mask[:, audio_token_length:] = attn_mask.clone()
        return shifted_input_ids, shifted_attn_mask
    
    def insert_promt(self, prompt, input_embeds, shifted_input_ids, shifted_attn_mask):
        if prompt:
            prompts = [prompt] * input_embeds.shape[0]
            # prompt_ids, prompt_mask (여기서는 중간에 추가하기때문에 special token을 추가할 필요가 없다.)
            prompts_ids, prompts_attn_mask = self.preprocess_text(prompts, input_embeds.device, add_special_tokens = False)
            # prompt_embeds -> input_embeds # 
            prompts_token_embeddings = self.get_decoder_embeddings()(prompts_ids)
            input_embeds = torch.cat((prompts_token_embeddings, input_embeds), dim = 1)
            prompts_ids[:,:]= -100
            # shifted_input_ids, shifted_mask
            shifted_input_ids = torch.cat((prompts_ids, shifted_input_ids), dim = 1)
            shifted_attn_mask = torch.cat((prompts_attn_mask, shifted_attn_mask), dim = 1)
        return input_embeds, shifted_input_ids, shifted_attn_mask
       
    def get_decoder_embeddings(self):
        if self.freeze_lm:
            return self.decoder.get_input_embeddings()
        else: # peft인 경우
            return self.decoder.base_model.get_input_embeddings() 
    
    def preprocess_text(self, caption, device, add_special_tokens = True):
        # pad, truncate, tensor return, add special token (sos, bos)
        tokenized_text = self.tokenizer(caption, padding = "longest", truncation = True, return_tensors = 'pt', add_special_tokens = add_special_tokens) # add_special_tokens = True
        input_ids = tokenized_text['input_ids'].to(device)
        attention_mask = tokenized_text['attention_mask'].to(device)   
        return input_ids, attention_mask
    
    def forward_decoder(self, proj_audio_embed, caption):
        # text input 준비 (tokenize -> ids, attention mask)
        input_ids, attn_mask = self.preprocess_text(caption, proj_audio_embed.device, add_special_tokens = True)
        
        # concat audio, text
        token_embeddings = self.get_decoder_embeddings()(input_ids)
        proj_audio_embed = proj_audio_embed.unsqueeze(1) # dimension 문제가 있을 것 같아서 늘림, clap에서 처리했나 확인하기
        input_embeds = torch.concat((proj_audio_embed, token_embeddings), dim = 1)
        
        # shifting & padding input ids
        shifted_input_ids, shifted_attn_mask = self.shift_tokenized_input(input_ids, attn_mask, proj_audio_embed.shape[1], proj_audio_embed.device)
        
        # inserting prompts: task_prompt
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_promt(self.task_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)

        # handling retrieved results
        # ..pass..
        
        # decoder forward pass (input embeds, shifted input ids, shifted attn mask)
        outputs = self.decoder(inputs_embeds = input_embeds, labels = shifted_input_ids, attention_mask = shifted_attn_mask)
        return outputs
        
    def forward(self, audio, caption):
        # audio encoder
        audio_embed = self.forward_encoder(audio)
        
        # align
        proj_audio_embed = self.forward_align(audio_embed)
        
        # decoder
        output = self.forward_decoder(proj_audio_embed, caption)
        
        # loss
        if output['loss'] is None:
            print("error: loss is None!")
        
        return output
    
    def generate_caption_inference(self, audio, caption):
        
        pass
    
if __name__ == "__main__":
    def get_config(args):
        # 파일에서 config 가져오기에 좋다.
        config = OmegaConf.load(args.config)
        return config
    
    parser = argparse.ArgumentParser(description="train_config")
    parser.add_argument("--config", type = str, default = "configs/train.yaml")
    args = parser.parse_args()
    
    # config 파일 불러오기
    config = get_config(args)
    
    # audio_data, _ = librosa.load('./examples/Yb0RFKhbpFJA.flac', sr=48000)
    # text_data = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    # audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    # audio_data = torch.tensor(audio_data).to("cuda")
    
    audio_caption_model = CLAP2LLAMA(config) # .to("cuda")
    # output = audio_caption_model(audio_data, text_data)
    # print(f"loss : {output['loss']}")
    # print(f"logits : {output['logits'].shape}")  # logits : torch.Size([1, 19, 32000])

    # captions = audio_caption_model.generate_caption(audio_data)
    # print(f"captions : {captions}")
