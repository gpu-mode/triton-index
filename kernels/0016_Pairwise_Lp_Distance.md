# Pairwise Lp Distance

**Link:** https://github.com/jinensetpal/triton_cdist/blob/3305c5592c51b51f1080933e58ab66c1fbaa620d/triton_cdist/lp_reduce.py

**Author:** Jinen Setpal

**Tags:** Distance

**Description:** A direct replacement of torch's [cdist](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html#torch-cdist), including support for backprop. Doesn't take torch's `compute_mode` argument.

**Minimal Usage:**
```py
import triton_cdist  # registers operator

# from here you can use it as a stand-in replacement of `torch.cdist`.

x1 = ...
x2 = ...
p = ...
torch.ops.triton_cdist.opt_cdist(x1, x2, p=p)  #  previously, `torch.cdist(x1, x1, p=p)`
```

**Triton Version:** v3.3.1+

**Id in triton index:** 0016
