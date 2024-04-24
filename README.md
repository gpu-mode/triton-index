# triton-index
Cataloging released Triton kernels.

This repo is meant to collect, predominantly, community-written and published Triton kernels. Triton can often be easier to understand or to start writing for a Python programmer only experienced with PyTorch, and it has recently seen an uptick in its usage--there is more Triton code around the web than there was before! This repo attempts to collect these kernels in one place, for people looking to learn Triton, seeing if a similar kernel already exists, or more.

## Kernels




## Other Libraries + Resources

- [The Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) are a great place to get started learning Triton.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops/triton) has a number of useful Triton kernels.
- [Unsloth](https://github.com/unslothai/unsloth) contains many ready-to-use Triton kernels especially for finetuning applications
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) has a massive number of Linear attention or subquadratic attention replacement architectures, written using several different approaches of parallelization in Triton.


## Contributing

To add a new entry, follow this template:


**Descriptive Entry Title/Name**
- Github Repo: (link to repository)
- Author: (contributor of kernel, if can be located)
- Link to kernel: (direct link to file containing kernel definition?)
- Triton Version: (e.g., "Triton v2.1.0")
- Other Notes: (e.g. "Useful example to reference of a quantization kernel", "cleanly commented, good for learning purposes", "Optimized for H100", ...)


