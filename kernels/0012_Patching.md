# Batched Patching

**Link:** https://github.com/cmeraki/vit.triton/blob/main/vit/kernels/patching.py

**Author:** Romit

**Tags:** Rearranging

**Description:** Implements batched patching of the input image. For a batch of images (B, C, H, W), the image is split into grids of (P, P) where P is the patch size. The patch is flattened into a 1D array of size (`P*P*C`).

**Minimal Usage:**

```python
import torch
from vit.kernels import patching

device = 'cuda'
dtype = torch.float32

batch_size = 4
height = 256
width = 256
channels = 3
patch_size = 16

print(f'Batch size: {batch_size}, Height: {height}, width: {width}, channels: {channels}, P: {patch_size}')

A = torch.arange(1, batch_size * height * width * channels + 1, dtype=dtype, device=device).view(batch_size, channels, height, width)

patches_triton = patching(A, patch_size)

print(f"Output size: {patches_triton.shape}")

print(f'Original matrix:\n{A}')
print(f'Triton patching:\n{patches_triton}')

```

**Triton Version:** v2.3.0

**Other Notes:**<br/>
Currently supports C = 3 only.

**Id in triton index:** 0012
