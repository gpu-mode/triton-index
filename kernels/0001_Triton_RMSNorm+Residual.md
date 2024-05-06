# Triton RMSNorm + Residual

**Link:** https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm.py

**Author:** Tri Dao

**Tags:** Norm

**Description:** <br/>Performs RMSNorm/LayerNorm on an input while also optionally adding a residual connection simultaneously. Supports row sizes up to 65536 // (bytes per element).

**Minimal Usage:**
```py
from mamba_ssm.ops.triton.layernorm import rms_norm_fn, layer_norm_fn
out = rms_norm_fn(x, weight, bias, residual=residual, residual_in_fp32=True, eps=1e-6)
``` 
**Triton Version:** v2.1.0+

**Other Notes:**<br/>Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

**Id in triton index:** 0001
