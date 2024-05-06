# Flash-Decoding / Split-KV Attention

**Link:** https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/triton_splitk.py

**Author:** Daniel Haziza, others?

**Tags:** Attention, Decoding

**Description:** <br/>A Triton kernel for performing attention while additionally parallelizing over the sequence dimension of the keys and values. Useful for fast, low-batch, long-context decoding. Very feature-rich--includes support for paged and/or quantized KV caches.

**Triton Version:** Triton v2.1.0+

**Other Notes:**<br/>Accompanied the release of the [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) blogpost. See it for more details and an explanation.

**Id in triton index:** 0002
