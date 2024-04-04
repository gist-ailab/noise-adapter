from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x

    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])


    # def state_dict(self, destination, prefix, keep_vars):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #     keys = [k for k in state.keys() if "rein" not in k]
    #     for key in keys:
    #         state.pop(key)
    #         if key in destination:
    #             destination.pop(key)
    #     return state
