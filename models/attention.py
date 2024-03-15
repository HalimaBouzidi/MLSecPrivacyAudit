
import torch, argparse
import numpy as np
from fightingcv_attention.attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.PSA import PSA
from fightingcv_attention.attention.EMSA import EMSA
from fightingcv_attention.attention.CoAtNet import CoAtNet
from fightingcv_attention.attention.HaloAttention import HaloAttention
from fightingcv_attention.attention.MobileViTAttention import MobileViTAttention
from fightingcv_attention.attention.MobileViTv2Attention import MobileViTv2Attention
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.CrissCrossAttention import CrissCrossAttention
from fightingcv_attention.attention.MUSEAttention import MUSEAttention
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.S2Attention import S2Attention

from fightingcv_attention.mlp.sMLP_block import sMLPBlock
from fightingcv_attention.mlp.mlp_mixer import MlpMixer
from fightingcv_attention.mlp.resmlp import ResMLP
from fightingcv_attention.mlp.g_mlp import gMLP


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_type", type=str, help="Type of the attention mechanim")
    parser.add_argument("--mlp_type", type=str, help="Type of the MLP mechanim")
    args = parser.parse_args()
    
    attn_type = args.attn_type
    mlp_type = args.mlp_type

    input=torch.randn(50,49,512)

    if attn_type == 'ext_attn': # same size as input
        ea = ExternalAttention(d_model=512,S=8)
        output=ea(input)
        print(output.shape)

    elif attn_type == 'self_attn': # same size as input
        sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
        output=sa(input,input,input)
        print(output.shape)

    elif attn_type == 'se_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        se = SEAttention(channel=512,reduction=8)
        output=se(input)
        print(output.shape) 

    elif attn_type == 'sk_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        se = SKAttention(channel=512,reduction=8)
        output=se(input)
        print(output.shape)

    elif attn_type == 'cbam_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        se = SKAttention(channel=512,reduction=8)
        output=se(input)
        print(output.shape)

    elif attn_type == 'bam_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        bam = BAMBlock(channel=512,reduction=16,dia_val=2)
        output=bam(input)
        print(output.shape)

    elif attn_type == 'eca_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        eca = ECAAttention(kernel_size=3)
        output=eca(input)
        print(output.shape)

    elif attn_type == 'pyramid_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        print(input.size())
        psa = PSA(channel=512,reduction=8)
        output=psa(input)
        print(output.shape)

    elif attn_type == 'efficient_attn': # same size as input
        emsa = EMSA(d_model=512, d_k=512, d_v=512, h=7,H=7,W=7,ratio=2,apply_transform=True)
        output=emsa(input,input,input)
        print(output.shape)

    elif attn_type == 'muse_attn': # same size as input
        sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
        output=sa(input,input,input)
        print(output.shape)

    elif attn_type == 'outlook_attn': # same size as input
        input = input.view(input.size(0), int(np.sqrt(input.size(1))), int(np.sqrt(input.size(1))), input.size(2))        
        outlook = OutlookAttention(dim=512)
        output=outlook(input)
        print(output.shape)

    elif attn_type == 'coat_attn': # not same size as input
        input=torch.randn(1,3,32,32)
        mbconv=CoAtNet(in_ch=3,image_size=32)
        out=mbconv(input)
        print(out.shape)

    elif attn_type == 'halo_attn': # same size as input
        input=torch.randn(1,512,8,8)
        halo = HaloAttention(dim=512,
            block_size=2,
            halo_size=1,)
        output=halo(input)
        print(output.shape)

    elif attn_type == 's2_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        s2att = S2Attention(channels=512)
        output=s2att(input)
        print(output.shape)

    elif attn_type == 'mb_attn': # same size as input
        m=MobileViTAttention(in_channel=3,dim=512,kernel_size=3,patch_size=7)
        input=torch.randn(1,3,49,49)
        output=m(input)
        print(output.shape) 

    elif attn_type == 'mb2_attn': # same size as input
        m=MobileViTv2Attention(d_model=512)
        output=m(input)
        print(output.shape) 

    elif attn_type == 'criss_attn': # same size as input
        input = input.transpose(2, 1) 
        input = input.view(input.size(0), input.size(1), int(np.sqrt(input.size(2))), int(np.sqrt(input.size(2))))
        model = CrissCrossAttention(in_dim=512)
        outputs = model(input)
        print(outputs.shape)

    else: 
        raise NotImplementedError

    
    if mlp_type == 'mix_mlp':
        input=torch.randn(50,3,40,40)
        mlp_mixer=MlpMixer(num_classes=1000,num_blocks=10,patch_size=10,tokens_hidden_dim=32,channels_hidden_dim=1024,tokens_mlp_dim=16,channels_mlp_dim=1024)
        output=mlp_mixer(input)
        print(output.shape) # output: torch.Size([50, 1000])

    elif mlp_type == 'res_mlp':
        input=torch.randn(50,3,14,14)
        resmlp=ResMLP(dim=128,image_size=14,patch_size=7,class_num=1000)
        out=resmlp(input)
        print(out.shape) # output: torch.Size([50, 1000])

    elif mlp_type == 'g_mlp':
        input=torch.randint(512,(50,49)) #bs,len_sen
        gmlp = gMLP(num_tokens=10000,len_sen=49,dim=512,d_ff=1024)
        output=gmlp(input)
        print(output.shape) # output: torch.Size([50, 49, 10000])

    elif mlp_type == 's_mlp':
        input=torch.randn(50,3,224,224)
        smlp=sMLPBlock(h=224,w=224)
        out=smlp(input)
        print(out.shape) # output: torch.Size([50, 3, 224, 224])

    else: 
        raise NotImplementedError