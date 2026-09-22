"""The three backbones of the benchmark -- taken OFF THE SHELF, not re-written.

    name        source                              params (C10 / C100)
    ---------   ---------------------------------   -------------------
    resnet18    torchvision.models.resnet18         11.17M / 11.22M
    resnet34    torchvision.models.resnet34         21.28M / 21.33M
    wrn28_10    pytorchcv `wrn28_10_cifar{10,100}`  36.48M / 36.54M
                (osmr/imgclsmob, the reference WRN of
                 Zagoruyko & Komodakis 2016)

WEIGHTS ARE ALWAYS RANDOM (`pretrained=False`).  This benchmark compares the
CONVERGENCE of optimizers, so every method must start from the same untrained
network; loading ImageNet or CIFAR weights would start every run near a solution
and measure fine-tuning instead of optimisation.  Only the architecture comes
from the library.

WHY WRN COMES FROM `pytorchcv` AND NOT torchvision: torchvision has no
WRN-28-10.  Its `wide_resnet50_2` / `wide_resnet101_2` are ImageNet bottleneck
nets (69M / 127M params), a different architecture entirely.  `pytorchcv` ships
the real CIFAR WRN-28-10 -- depth 28, widen 10, pre-activation blocks -- and the
parameter counts above match the paper.  `pip install pytorchcv` (already a
requirement of this folder).

WHY THE ResNets NEED A STEM SWAP: torchvision's ResNet is the ImageNet one, with
a 7x7 stride-2 stem plus a 3x3 stride-2 max-pool, which reduces a 32x32 CIFAR
image to 8x8 before the first residual block and costs ~5 accuracy points.
Every CIFAR ResNet number in the literature uses the variant with a 3x3 stride-1
stem and no max-pool, which is what `cifar_stem=True` (the default) installs.
This changes no weights that matter -- the stem is re-initialised either way --
it only changes the spatial resolution the blocks see.

    from models import make_model
    net = make_model("wrn28_10", num_classes=100)
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

__all__ = ["make_model", "make_groups", "MODELS", "n_params", "param_norm",
           "model_source"]

MODELS = ("resnet18", "resnet34", "wrn28_10")

SOURCE = {"resnet18": "torchvision.models.resnet18",
          "resnet34": "torchvision.models.resnet34",
          "wrn28_10": "pytorchcv:wrn28_10_cifar{n}"}


def model_source(name: str) -> str:
    return SOURCE[name]


def _torchvision_resnet(depth: int, num_classes: int, cifar_stem: bool) -> nn.Module:
    from torchvision import models as tvm

    net = {18: tvm.resnet18, 34: tvm.resnet34}[depth](weights=None,
                                                      num_classes=num_classes)
    if cifar_stem:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out",
                                nonlinearity="relu")
    return net


def _pytorchcv_wrn(num_classes: int) -> nn.Module:
    try:
        from pytorchcv.model_provider import get_model
    except ImportError as e:                      # pragma: no cover
        raise ImportError(
            "wrn28_10 comes from pytorchcv (torchvision has no WRN-28-10). "
            "Install it with `pip install pytorchcv`.") from e
    if num_classes not in (10, 100):
        raise ValueError("pytorchcv ships WRN-28-10 for 10 or 100 classes, "
                         "got num_classes=%d" % num_classes)
    return get_model("wrn28_10_cifar%d" % num_classes, pretrained=False)


def make_model(name: str, num_classes: int, *, cifar_stem: bool = True,
               **kw) -> nn.Module:
    """An untrained network of `MODELS`, sized for `num_classes`."""
    if name not in MODELS:
        raise KeyError("unknown model %r; have %s" % (name, list(MODELS)))
    if kw:
        raise TypeError("unexpected model kwargs %s" % list(kw))
    if name == "wrn28_10":
        return _pytorchcv_wrn(num_classes)
    return _torchvision_resnet(int(name[6:]), num_classes, cifar_stem)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_norm(params) -> float:
    import torch
    ps = list(params)
    if not ps:
        return 0.0
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in ps)))


def make_groups(model: nn.Module, block: str = "module", radius: float = 2.0,
                radius_mode: str = "rel") -> List[Dict]:
    """The blocks of C = prod_g C_g, one torch param_group each.

    block = "all"     one ball for the whole weight vector (the literal reading
                      of the paper: C is ONE compact convex set)
            "module"  one ball per leaf module (conv / bn / linear) -- the
                      layer-wise constraint used by the deep Frank-Wolfe
                      literature, and the default here
            "param"   one ball per tensor (the finest product set)

    radius_mode = "rel" sets R_g = radius * ||theta_g^0||, a radius measured in
    units of the initialisation norm of that block.  That is the only scale that
    transfers across resnet18 (11M), resnet34 (21M) and wrn28_10 (36M): with
    R = 1 the initialisation sits exactly ON the boundary, with R = 2 it has room
    to grow.  radius_mode = "abs" uses the same R for every block, which is what
    the MARL code did (equally sized actors).
    """
    if block == "all":
        named = [("all", [p for p in model.parameters() if p.requires_grad])]
    elif block == "module":
        named = []
        for name, m in model.named_modules():
            ps = [p for p in m.parameters(recurse=False) if p.requires_grad]
            if ps:
                named.append((name or "root", ps))
    elif block == "module_nobn":
        # Like "module" but BatchNorm layers are left OUTSIDE C (unconstrained).
        # BatchNorm's scale/bias (gamma/beta) are updated by the FW step as usual
        # but their group is omitted here, so they are never projected or
        # constrained -- the FW update still moves them, just without a radius.
        # Use this when you want the constraint to cover only conv/linear weights.
        named = []
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                continue   # skip BatchNorm -- left unconstrained
            ps = [p for p in m.parameters(recurse=False) if p.requires_grad]
            if ps:
                named.append((name or "root", ps))
    elif block == "param":
        named = [(n, [p]) for n, p in model.named_parameters() if p.requires_grad]
    else:
        raise KeyError(
            "unknown block %r; have ['all', 'module', 'module_nobn', 'param']" % block
        )

    groups: List[Dict] = []
    for name, ps in named:
        g: Dict = {"params": ps, "name": name}
        if radius_mode == "rel":
            g["radius"] = max(radius * param_norm(ps), 1e-8)   # zero-init biases
        elif radius_mode != "abs":
            raise KeyError("unknown radius_mode %r; have ['rel', 'abs']" % radius_mode)
        groups.append(g)
    return groups
