# triton-index

Goals of this repo:
- **Catalog openly available Triton kernels**, so
    - (i) practitioners save work and
    - (ii) learners have world-class examples to learn from.
- **Surface which Triton kernels are still needed by the community**, so
    -  (i) work of the community can be more targeted, and
    - (ii) eager people new to our community have projects to demonstrate their skill.

Triton is easier to understand and start with than CUDA, especially for Python programmers experienced with PyTorch. And it has recently seen an uptick in its usage -- there is more Triton code around the web than there was before!

This repo collects these kernels in one place, for the benefit of practitioners and learners in our community.

Contributions are very, very welcome!

## Kernels

- [Here's the link](kernel_overview.md) of an ctrl-f searchable overview
- The kernel entries are in the [/kernels directory](/kernels/)

## Other Libraries and Resources

- [A Practioner's Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24&t=93s) is a great gentle intro to Triton (here's the [accompanying notebook](https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb)).
    - _Note from Umer: It feels weird to include my own work, but [many](https://twitter.com/jeremyphoward/status/1781438834561126776) [people](https://twitter.com/marksaroufim/status/1778978773721092274) said the lecture was very helpful!_

- [The Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) are a great intermediate resources.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops/triton) has a number of useful Triton kernels.
- [Unsloth](https://github.com/unslothai/unsloth) contains many ready-to-use Triton kernels especially for finetuning applications
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) has a massive number of Linear attention or subquadratic attention replacement architectures, written using several different approaches of parallelization in Triton.
- [xformers](https://github.com/facebookresearch/xformers/) contains many [Triton kernels](https://github.com/facebookresearch/xformers/tree/main/xformers/triton) throughout, including some [attention kernels](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py) such as [Flash-Decoding](https://pytorch.org/blog/flash-decoding/).


## Contributing

To add a new entry:
1. Copy [kernels/0000_template.md](kernels/0000_template.md) and fill it out
2. Add your entry to [kernel_overview.md](kernel_overview.md)
3. That's it!

<br/>

---

Brought to you by the communtiy, initiated by [Hailey](https://x.com/haileysch__) and [Umer](https://x.com/UmerHAdil) ❤️
