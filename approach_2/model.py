from tkinter import Y
import torch
import torch.nn as nn
import torchvision.models as models


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

      
class VisionTransformerEmbedded(nn.Module):

    def __init__(self, hparams, embed_dim, hidden_dim, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        super().__init__()

        self.hparams=hparams
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.Resnets = [nn.Sequential(
            # add 1-2 convolutions here for potential downscaling of the patch into 224x224
            *(list(models.resnet18(pretrained=False).children())[:-2])
            )\
            for i in range(self.num_patches)]
        self.flattened_dim = 25088 #512x7x7

        # Layers/Networks
        self.input_layer = nn.Linear(self.flattened_dim, embed_dim)#num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))



    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #input is of (batch_size, num_patches, 3, patch_size, patch_size) dimensionality 
        x = x.permute(1,0,2,3,4)
        tmp = torch.zeros(self.num_patches, self.hparams['batch_size'], 512, 7, 7)
        for i in range(self.num_patches):
            #x[i] is of (8,3,224,224)
            tmp[i] = self.Resnets[i](x[i])
            #tmp[i] is of (8,512,7,7)
        x = tmp.permute(1,0,2,3,4)
        #x is of (8,4,512,7,7)
        x = x.flatten(2,4)
        #x is of (8, 4, 25088)

        ###################### ViT from here
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
