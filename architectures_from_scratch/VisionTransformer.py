import torch 
import torch.nn as nn 


class PatchEmbed(nn.Module): 
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).
    patch_size : int
        Size of the patch (it is a square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        The emmbedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans, 
            out_channels=embed_dim, 
            # kernel/stride == to patch size, feat maps don't overlap btween patches 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x) -> None: 
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """

        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)


class Attention(nn.Module):

    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.) -> None:
        super().__init__()
        self.n_heads = n_heads, 
        self.dim = dim, 
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5