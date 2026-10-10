# MineFormer v1

PyTorch implementation of a hybrid encoder and multi-scale decoder for binary
segmentation. No boundary head is included.

| Component | Default channels | Spatial scale |
| --- | --- | --- |
| CNN stem | 32 | 1/2 |
| CNN stage 1 | 64 | 1/4 |
| CNN stage 2 | 128 | 1/8 |
| Transformer stage 3 | 256 | 1/16 |
| Transformer stage 4 | 512 | 1/32 |
| Multi-scale fusion decoder | 128 | 1/4 |
| Binary segmentation head | 1 | Input resolution |

CNN stages use residual convolution blocks. Transformer stages use stride-2
convolutional embedding, depthwise convolutional positional encoding, and
pre-norm global self-attention with an MLP. All four encoder features are
projected, resized to stage 1 resolution, concatenated, and fused by convolutions.
Group normalization in the convolutional blocks supports small training batches.

These are configurable architectural defaults; no pretrained weights or training
recipe are provided. Global attention memory grows quadratically with spatial
token count, so choose image size and batch size to fit available memory.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

## Use

```python
import torch
from mineformer import MineFormerV1

model = MineFormerV1()
images = torch.randn(2, 3, 256, 256)
logits = model(images)  # (2, 1, 256, 256)
targets = torch.randint(0, 2, logits.shape).float()
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
loss.backward()
probabilities = logits.detach().sigmoid()
masks = probabilities >= 0.5
```

Inputs must be floating point NCHW tensors with height and width at least 32.
Odd and non-square dimensions are supported; logits always match input size.
The model returns raw logits, without sigmoid. Customize `in_channels`,
`channels`, `depths`, `heads`, `stem_channels`, `decoder_channels`, `mlp_ratio`,
and `dropout` through the constructor. `forward_features(images)` returns the
four encoder feature maps in stage order.
