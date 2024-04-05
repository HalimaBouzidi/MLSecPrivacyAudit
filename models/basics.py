import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.widths = [32, 64, 128, 256, 512]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = nn.Conv2d(3, self.expanded[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.expanded[0], self.expanded[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.expanded[1], self.expanded[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.expanded[2], self.expanded[3], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.expanded[3], self.expanded[4], kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(self.expanded[4], self.expanded[4]*2)  # 2*2 comes from image dimension reduction
        self.fc2 = nn.Linear(self.expanded[4]*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self.expanded[4])
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

def DepthwiseSeparableConv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class DepthCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(DepthCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = DepthwiseSeparableConv(3, self.expanded[0], 1)
        self.conv2 = DepthwiseSeparableConv(self.expanded[0], self.expanded[1], 1)
        self.conv3 = DepthwiseSeparableConv(self.expanded[1], self.expanded[2], 1)
        self.conv4 = DepthwiseSeparableConv(self.expanded[2], self.expanded[3], 1)

        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3]*2)
        self.fc2 = nn.Linear(self.expanded[3]*2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)

        x = x.view(-1, self.expanded[3] * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottleneckConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckConv, self).__init__()
        # 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 convolution to expand dimensions
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class BottleCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(BottleCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = BottleneckConv(3, int(self.expanded[0]/4), self.expanded[0])
        self.conv2 = BottleneckConv(self.expanded[0], int(self.expanded[1]/4), self.expanded[1])
        self.conv3 = BottleneckConv(self.expanded[1], int(self.expanded[2]/4), self.expanded[2])
        self.conv4 = BottleneckConv(self.expanded[2], int(self.expanded[3]/4), self.expanded[3])

        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3])
        self.fc2 = nn.Linear(self.expanded[3], num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        x = x.view(-1, self.expanded[3] * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 convolution to expand dimensions
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]

        self.block1 = BottleneckBlock(3, int(self.expanded[0]/4), self.expanded[0], stride=1)
        self.block2 = BottleneckBlock(self.expanded[0], int(self.expanded[1]/4), self.expanded[1], stride=2)  # Stride=2 for downsampling
        self.block3 = BottleneckBlock(self.expanded[1], int(self.expanded[2]/4), self.expanded[2], stride=2)  # Stride=2 for downsampling
        self.block4 = BottleneckBlock(self.expanded[2], int(self.expanded[3]/4), self.expanded[3], stride=2)  # Stride=2 for downsampling

        # Fully connected layers
        self.fc1 = nn.Linear(self.expanded[3] * 2, self.expanded[3])  # Adjusted for the downsampled image size
        self.fc2 = nn.Linear(self.expanded[3], num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global Average Pooling
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, self.expanded[3] * 2)  # Flatten the output for the fully connected layer

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AsymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(AsymmetricConv2d, self).__init__()
        # First asymmetric convolution: 1xN
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding))
        # Second asymmetric convolution: Nx1
        self.conv2 = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AsymmetricCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(AsymmetricCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        
        self.conv1 = AsymmetricConv2d(3, self.expanded[0], kernel_size=3, padding=1)
        self.conv2 = AsymmetricConv2d(self.expanded[0], self.expanded[1], kernel_size=3, padding=1)
        self.conv3 = AsymmetricConv2d(self.expanded[1], self.expanded[2], kernel_size=3, padding=1)
        self.conv4 = AsymmetricConv2d(self.expanded[2], self.expanded[3], kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3]*2)
        self.fc2 = nn.Linear(self.expanded[3]*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self.expanded[3] * 2 * 2)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

class PatchEmbedding(nn.Module): # Done
    """
    img_size: 1d size of each image (32 for CIFAR-10)
    patch_size: 1d size of each patch (img_size/num_patch_1d, 4 in this experiment)
    in_chans: input channel (3 for RGB images)
    emb_dim: flattened length for each token (or patch)
    """
    def __init__(self, img_size:int, patch_size:int, in_chans:int=3, emb_dim:int=48):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, 
            emb_dim, 
            kernel_size = patch_size, 
            stride = patch_size
        )

    def forward(self, x):
        with torch.no_grad():
            # x: [batch, in_chans, img_size, img_size]
            x = self.proj(x) # [batch, embed_dim, # of patches in a row, # of patches in a col], [batch, 48, 8, 8] in this experiment
            x = x.flatten(2) # [batch, embed_dim, total # of patches], [batch, 48, 64] in this experiment
            x = x.transpose(1, 2) # [batch, total # of patches, emb_dim] => Transformer encoder requires this dimensions [batch, number of words, word_emb_dim]
        return x


class TransformerEncoder(nn.Module): # Done
    def __init__(self, input_dim:int, mlp_hidden_dim:int, num_head:int=8, dropout:float=0.):
        # input_dim and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadSelfAttention(input_dim, n_heads=num_head)
        self.norm2 = nn.LayerNorm(input_dim)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.msa(self.norm1(x)) + x # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    dim: dimension of input and out per token features (emb dim for tokens)
    n_heads: number of heads
    qkv_bias: whether to have bias in qkv linear layers
    attn_p: dropout probability for attention
    proj_p: droupout probability last linear layer
    scale: scaling factor for attention (1/sqrt(dk))
    qkv: initial linear layer for the query, key, and value
    proj: last linear layer
    attn_drop, proj_drop: dropout layers for attn and proj
    """
    def __init__(self, dim:int, n_heads:int=8, qkv_bias:bool=True, attn_p:float=0.01, proj_p:float=0.01):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim # embedding dimension for input
        self.head_dim = dim // n_heads # d_q, d_k, d_v in the paper (int div needed to preserve input dim = output dim)
        self.scale = self.head_dim ** -0.5 # 1/sqrt(d_k)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # lower linear layers in Figure 2 of the paper
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim) # upper linear layers in Figure 2 of the paper
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input and Output shape: [batch_size, n_patches + 1, dim]
        """
        batch_size, n_tokens, x_dim = x.shape # n_tokens = n_patches + 1 (1 is cls_token), x_dim is input dim

        # Sanity Check
        if x_dim != self.dim: # make sure input dim is same as concatnated dim (output dim)
            raise ValueError
        if self.dim != self.head_dim*self.n_heads: # make sure dim is divisible by n_heads
            raise ValueError(f"Input & Output dim should be divisible by Number of Heads")
        
        # Linear Layers for Query, Key, Value
        qkv = self.qkv(x) # (batch_size, n_patches+1, 3*dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim) # (batch_size, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (batch_size, n_heads, n_patches+1, head_dim)

        # Scaled Dot-Product Attention
        k_t = k.transpose(-2, -1) # K Transpose: (batch_size, n_heads, head_dim, n_patches+1)
        dot_product = (q @ k_t)*self.scale # Query, Key Dot Product with Scale Factor: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = dot_product.softmax(dim=-1) # Softmax: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn) # Attention Dropout: (batch_size, n_heads, n_patches+1, n_patches+1)
        weighted_avg = attn @ v # (batch_size, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch_size, n_patches+1, n_heads, head_dim)

        # Concat and Last Linear Layer
        weighted_avg = weighted_avg.flatten(2) # Concat: (batch_size, n_patches+1, dim)
        x = self.proj(weighted_avg) # Last Linear Layer: (batch_size, n_patches+1, dim)
        x = self.proj_drop(x) # Last Linear Layer Dropout: (batch_size, n_patches+1, dim)

        return x

class SimpleViT(nn.Module): # Done
    def __init__(self, width_expand=1.0, num_classes:int=10, img_size:int=32, num_patch_1d:int=8, dropout:float=0., 
                 num_enc_layers:int=6, hidden_dim:int=96, mlp_hidden_dim:int=96*4, num_head:int=4, is_cls_token:bool=True):
        super(SimpleViT, self).__init__()
        """
        is_cls_token: are we using class token?
        num_patch_1d: number of patches in one row (or col), 3 in Figure 1 of the paper, 8 in this experiment
        patch_size: # 1d size (size of row or col) of each patch, 16 for ImageNet in the paper, 4 in this experiment
        flattened_patch_dim: Flattened vec length for each patch (4 x 4 x 3, each side is 4 and 3 color scheme), 48 in this experiment
        num_tokens: number of total patches + 1 (class token), 10 in Figure 1 of the paper, 65 in this experiment
        """
        hidden_dim = int(width_expand*hidden_dim)
        mlp_hidden_dim = int(width_expand*mlp_hidden_dim)
        
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d
        self.patch_size = img_size//self.num_patch_1d
        flattened_patch_dim = (img_size//self.num_patch_1d)**2*3
        num_tokens = (self.num_patch_1d**2)+1 if self.is_cls_token else (self.num_patch_1d**2)

        # Divide each image into patches
        self.images_to_patches = PatchEmbedding(
                                    img_size=img_size, 
                                    patch_size=img_size//num_patch_1d
                                )

        # Linear Projection of Flattened Patches
        self.lpfp = nn.Linear(flattened_patch_dim, hidden_dim) # 48 x 384 (384 is the latent vector size D in the paper)

        # Patch + Position Embedding (Learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None # learnable classification token with dim [1, 1, 384]. 1 in 2nd dim because there is only one class per each image not each patch
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)) # learnable positional embedding with dim [1, 65, 384]
        
        # Transformer Encoder
        enc_list = [TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head) for _ in range(num_enc_layers)] # num_enc_layers is L in Transformer Encoder at Figure 1
        self.enc = nn.Sequential(*enc_list) # * should be adeed if given regular python list to nn.Sequential
        
        # MLP Head (Standard Classifier)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x): # x: [batch, 3, 32, 32]
        # Images into Patches (including flattening)
        out = self.images_to_patches(x) # [batch, 64, 48]

        # Linear Projection on Flattened Patches
        out = self.lpfp(out) # [batch, 64, 384]

        # Add Class Token and Positional Embedding
        if self.is_cls_token: 
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1) # [batch, 65, 384], added as extra learnable embedding
        out = out + self.pos_emb # [batch, 65, 384]

        # Transformer Encoder
        out = self.enc(out) # [batch, 65, 384]
        if self.is_cls_token:
            out = out[:,0] # [batch, 384]
        else:
            out = out.mean(1)

        # MLP Head
        out = self.mlp_head(out) # [batch, 10]
        return out

# class SimpleViT(nn.Module):
#     def __init__(self, in_channels=3, width_expand=1.0, num_classes=10, depth=6):
#         super(SimpleViT, self).__init__()        
#         width = int(64*width_expand)
#         self.transformer = models.VisionTransformer(image_size=32, patch_size=4, num_layers=depth, \
#                                 hidden_dim=width, mlp_dim=width, num_heads=8, num_classes=num_classes)
#         self.transformer.conv_proj = nn.Conv2d(in_channels, width, kernel_size=(4, 4), stride=(4, 4))

#     def forward(self, x):
#         return self.transformer(x)

def get_model(args):

    if args['train']['model_name'] == "simple_cnn":
        return SimpleCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "depth_cnn":
        return DepthCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "bottle_cnn":
        return BottleCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "resnet_cnn":
        return SimpleResNet(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "asymetric_cnn":
        return AsymmetricCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "simple_vit":
        return SimpleViT(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    else:
        raise NotImplementedError(f"{args['train']['model_name']} is not implemented")

if __name__ == '__main__':
    
    widths = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for width in widths:
        model = SimpleCNN(width_expand=width, num_classes=100)
        print(width, model(torch.rand(1, 3, 32, 32)).size())