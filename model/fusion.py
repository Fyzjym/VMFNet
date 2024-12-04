import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat
import math
from model.resnet_dilation import resnet18 as resnet18_dilation
from model.transformer import *

### mix the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(Mix_TR, self).__init__()
        
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.style_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, style_norm)

        fre_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, fre_norm)

        cont_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.cont_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, cont_norm)


        ### fusion the content and style in the transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        fre_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, fre_decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        # self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) # add 1D position encoding
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding

        # self.high_pro_mlp = nn.Sequential(
        #     nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        # self.low_pro_mlp = nn.Sequential(
        #     nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_feature_filter = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        self._reset_parameters()

        ### low frequency style encoder
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        ### hig frequency style encoder
        self.freq_encoder = self.initialize_resnet18()
        self.freq_dilation_layer = resnet18_dilation().conv5_x

        ### content encoder
        # self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))
        self.content_encoder = self.initialize_resnet18()
        self.content_dilation_layer = resnet18_dilation().conv5_x


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def process_style_feature(self, encoder, dilation_layer, style, add_position2D, style_encoder):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=16).contiguous()
        style = dilation_layer(style)
        style = add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        style = style_encoder(style)
        return style

    
    def get_low_style_feature(self, style):
        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, style, self.add_position2D, self.style_encoder)

    def get_high_style_feature(self, laplace):
        return self.process_style_feature(self.freq_encoder, self.freq_dilation_layer, laplace, self.add_position2D, self.fre_encoder)

    def get_content_style_feature(self, content):
        return self.process_style_feature(self.content_encoder, self.content_dilation_layer, content, self.add_position2D, self.cont_encoder)


    def forward(self, style, laplace, content):

        # get the highg frequency and style feature
        laplace = self.get_high_style_feature(laplace)

        # get the low frequency and style feature
        style = self.get_low_style_feature(style)
        mask = self.low_feature_filter(style)
        style = style * mask

        # content encoder
        content = self.get_content_style_feature(content)

        # fusion of content and style features
        style_hs = self.decoder(content, style, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], laplace, tgt_mask=None)

        hs = hs[0].permute(1, 0, 2).contiguous()
        hs = rearrange(hs, 'n c (h w) -> n w c h', h=256, w=2)

        return hs