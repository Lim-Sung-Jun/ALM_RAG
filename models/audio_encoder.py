from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from laion_clap import CLAP_Module
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

# 이렇게 선언해주면 audio_encoder의 config로 들어가게 된다.
class CLAPEncoderConfig(PretrainedConfig):
    model_type = "audio_encoder"
    def __init__(self,
                 model_name: str = "CLAPAudioEncoder",
                 pretrained: bool = True,
                 freeze: bool = True,
                 spec_augment: bool = True,
                 audio_args: dict = True,
                 select_feature = "embedding",
                 **kwargs): 
                # kwargs?
        # super(CLAPEncoderConfig, self).__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.select_feature = select_feature # fine-grained, embedding, projected
        super(CLAPEncoderConfig, self).__init__(**kwargs)
        
        ### ?
        # self.sequence_length = 1024
        # self.hidden_size = 768
        # self.window_size = 4 # (32,32) = [B,32,H], (16,16) = [B,64,H], (8,8) = [B,128,H] (4,4) = [B,256,H]
        # self.step_size = 4
    
class CLAPAudioTower(PreTrainedModel):
    config_class = CLAPEncoderConfig
    
    def __init__(self, config):
        super(CLAPAudioTower, self).__init__(config)
        self.config = config
        
        # 데이터 preprocessing에 이미 audio_feature를 추출하는데 clap forward를 하면 이미 그걸 지나는 것 같은데? 확인해보자!
        self.clap = CLAP_Module(enable_fusion= True, device = torch.device(self.config.device))
        if config.pretrained == True:
            self.clap.load_ckpt()
            
        # modified function: 밑에는 512차원 768차원이 있는데 차이가 .. .음 llava나 pengi는 clap이 projection이 된건가? 이거를 512차원으로 바꾸면 안되나? 나중에 512로 바꿔보자.
        def get_audio_embedding_patch(self, data,
                                      select_feature=self.config.select_feature,
                                      window_size=self.config.window_size,
                                      step_size=self.config.step_size):
            device = next(self.parameters()).device
            input_dict = {}
            keys = data[0].keys()
            for k in keys:
                input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
            audio_embeds = self.encode_audio(input_dict, device=device)
            audio_embeds = audio_embeds[select_feature]
            audio_embeds = F.normalize(audio_embeds, dim=-1)
            if select_feature == "fine_grained_embedding":
                embeds = audio_embeds[select_feature] # [B,1024,768]
                unfolded = embeds.unfold(1, window_size, step_size) # [B,1024/S,768,W]
                averaged = unfolded.mean(dim=-1) # [B,1024/S,768]
                return averaged
            else:
                return audio_embeds
        self.clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_patch, self.clap.model)
        
        # original function
        # def get_audio_embedding(self, data):
        #     """Get the audio embedding from the model

        #     Parameters
        #     ----------
        #     data: a list of dict
        #         the audio input dict list from 'get_audio_feature' method

        #     Returns
        #     ----------
        #     audio_embed: torch.Tensor
        #         a tensor of audio_embeds (N, D)

        #     """
        #     device = next(self.parameters()).device
        #     input_dict = {}
        #     keys = data[0].keys()
        #     for k in keys:
        #         input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        #     audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
        #     audio_embeds = self.audio_projection(audio_embeds)
        #     audio_embeds = F.normalize(audio_embeds, dim=-1)
        #     return audio_embeds

    @torch.no_grad()
    def forward(self, x):
        # x는 list of temp_dict(preprocessed audio feature)
        outputs = self.clap.model.get_audio_embedding(x)
        # hidden_states, attention weights
        return BaseModelOutput(outputs, None, None)