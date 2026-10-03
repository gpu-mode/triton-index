# Fused Cross Entropy (forward + backward, in-place gradient)

**Link:** https://github.com/dh8116/triton-kernels/blob/main/fused_cross_entropy.py

**Author:** Richael (dh8116)

**Tags:** loss, cross entropy, fused, memory

**Description:** <br/>
- Computes the cross-entropy loss and its gradient in a single pass. dL/dlogits = softmax(logits) - onehot(target) depends only on the logits, so the gradient is computed during the forward pass and written back over the input buffer, and no second [N, V] tensor is allocated.
- Supports `ignore_index` and a custom mean denominator (`n_valid`).
- Measured on a T4, 2048 rows, vocab 131,072, fp16, against a fused fp16 `F.cross_entropy`: 24.03 ms -> 15.90 ms (1.51x), peak memory 2684 MB -> 1611 MB (1.67x).
- Known limitation, documented in the file: in fp16 at vocab 32k, 74.3% of stored gradient values underflow to zero (each is ~1/(vocab*batch), below fp16's smallest subnormal). fp32 is fine.

**Minimal Usage:**
```py
from fused_cross_entropy import fused_ce

logits = torch.randn(2048, 131072, device="cuda", dtype=torch.float16, requires_grad=True)
targets = torch.randint(0, 131072, (2048,), device="cuda")
loss = fused_ce(logits, targets)   # logits' storage is reused for the gradient
loss.backward()
```

**Triton Version:** 3.x

**Other Notes:**<br/>Write-up with the benchmark method: https://dh8116.github.io/blog/fused-cross-entropy. A useful learning example of trading the input buffer for the gradient. Note that the logits are overwritten, so don't reuse them after the call.

**Id in triton index:** 0016
