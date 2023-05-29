import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip

from config import Config


def load_cache():
    load_open_clip_cache()
    load_timm_cache()

def load_open_clip_cache():
    _model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', 'laion2b_s32b_b82k')  # これだけnormalizeが違うので注意Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    _model, _, _ = open_clip.create_model_and_transforms('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    _model, _, _ = open_clip.create_model_and_transforms('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind')

def load_timm_cache():
    _model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k', pretrained=True)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


def get_norm_layer(norm_name):
    if norm_name == 'batch':
        return nn.BatchNorm1d
    elif norm_name == 'layer':
        return LayerNorm
    else:
        raise NotImplementedError(f"invalid norm_name={norm_name}")


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)

class GeMP(nn.Module):
    def __init__(self, p=3, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self._learn_p = learn_p
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        x = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

        x = x.view(x.size(0), -1)
        return x


class GGeMP(nn.Module):
    def __init__(self, num_groups, p=5, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(num_groups) * p)
        self.num_groups = num_groups
        self.learn_p = learn_p
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag: bool):
        self.learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape((B, N, D // self.num_groups, self.num_groups))

        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)
        x = x.pow(1.0 / self.p)

        x = x.reshape((B, D))
        return x

def gen_pooling(conf):
    if conf.pooling == 'avg':
        global_pool = torch.nn.AdaptiveAvgPool2d(1)
    elif conf.pooling == 'gem':
        global_pool = GeMP(p=conf.gemp_p, learn_p=conf.gemp_learn_p)
    elif conf.pooling == 'ggemp':
        global_pool = GGeMP(num_groups=conf.ggemp_num_groups, p=conf.ggemp_p, learn_p=conf.ggemp_learn_p)
    elif conf.pooling == 'max':
        global_pool = torch.nn.AdaptiveMaxPool2d(1)
    elif conf.pooling == 'avgmax':
        global_pool = AdaptiveAvgMaxPool2d()
    elif conf.pooling == 'catavgmax':
        global_pool = AdaptiveCatAvgMaxPool2d()
    elif conf.pooling == 'nop':
        global_pool = nn.Identity()
    else:
        raise NotImplementedError(f'Invalid pooling type: {conf.pooling}')

    return global_pool


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


# timmのViTモデル向け
class TimmViTModel(nn.Module):
    def __init__(
        self,
        conf: Config,
        backbone: nn.Module,
        n_features: int,
    ):
        """Initialize"""
        super().__init__()
        self.h_patch = conf.h_patch
        self.w_patch = conf.w_patch
        self.backbone = backbone
        self.n_features = n_features
        self.with_cls = conf.with_cls
        self.cls_mode = conf.cls_mode
        self.pooling = conf.pooling

        self.global_pool = gen_pooling(conf)
        if conf.pooling == 'catavgmax':
            self.n_features *= 2

        if conf.with_cls and (conf.cls_mode == 'cat'):
            self.fc = nn.Linear(self.n_features*2, 384, bias=(not conf.no_fc_bias))
        else:
            self.fc = nn.Linear(self.n_features, 384, bias=(not conf.no_fc_bias))

        self._init_params()

    @torch.jit.ignore
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if type(self.fc.bias) == torch.nn.parameter.Parameter:
            nn.init.constant_(self.fc.bias, 0)
            nn.init.constant_(self.normalize.weight, 1)
            nn.init.constant_(self.normalize.bias, 0)

    def forward(self, x):
        """Forward"""
        org_feat = self.backbone.forward_features(x)

        reshaped = False
        if len(org_feat.shape) == 3:
            bs, _, dim = org_feat.shape
        elif len(org_feat.shape) == 4:
            bs, _h_patch, _w_patch, dim = org_feat.shape
            org_feat = org_feat.permute(0, 3, 1, 2)
            reshaped = True
        else:
            raise

        feat = torch.zeros(1, dtype=org_feat.dtype, device=org_feat.device)
        if self.with_cls:  # clsも使う
            cls = org_feat[:,0,:]  # bs, dim

            if self.cls_mode in ['avg', 'cat', 'without_cls']:
                if reshaped:
                    feat = org_feat
                else:
                    feat = org_feat[:,1:,:]
                    feat = feat.permute(0,2,1).reshape(bs, dim, self.h_patch, self.w_patch)
        else:  # clsを使わない / clsが無い
            cls = torch.zeros(1, dtype=org_feat.dtype, device=org_feat.device)
            if reshaped:
                feat = org_feat
            else:
                feat = org_feat.permute(0,2,1).reshape(bs, dim, self.h_patch, self.w_patch)

        if self.cls_mode in ['avg', 'cat', 'without_cls']:
            feat = self.global_pool(feat)
            feat = feat.view(feat.size(0), -1)  # bs, dim

        if self.with_cls:
            if self.cls_mode == 'avg':
                feat = (cls + feat) / 2.
            elif self.cls_mode == 'cat':
                feat = torch.cat([cls, feat], dim=1)
            elif self.cls_mode == 'cls_only':
                feat = cls
            elif self.cls_mode == 'without_cls':
                # featをそのまま使う
                pass
            elif self.cls_mode == 'full_avg':
                # cls含めて全体の平均を使う
                feat = org_feat.mean(dim=1)
            else:
                raise NotImplementedError(f"invalid cls_mode: {self.cls_mode}")
        else:
            pass

        feat = self.fc(feat)
        feat = F.normalize(feat)

        return feat


class ViTHeadlessModel(nn.Module):
    def __init__(
        self,
        conf: Config,
        backbone: nn.Module,
        n_features: int,
    ):
        """Initialize"""
        super().__init__()
        self.backbone = backbone
        self.n_features = n_features

        self.fc = nn.Linear(self.n_features, 384, bias=(not conf.no_fc_bias))

        self._init_params()


    @torch.jit.ignore
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if type(self.fc.bias) == torch.nn.parameter.Parameter:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """Forward"""
        feat = self.backbone(x)
        feat = self.fc(feat)
        feat = F.normalize(feat)

        return feat



model_dict = {
    'ViTHeadlessModel': ViTHeadlessModel,
    'TimmViTModel': TimmViTModel,
}


def get_model(conf, logger):

    backbone_name = conf.backbone_name

    # create backbone
    if backbone_name in ['ViT-L-14']:
        model, _, _ = open_clip.create_model_and_transforms(backbone_name, pretrained=conf.backbone_pretrained)
        backbone = model.visual
        if conf.set_grad_checkpointing:
            backbone.set_grad_checkpointing(True)
            logger.info('use Gradient Checkpointing')
        n_features = backbone.output_dim

        if conf.input_width != 224:
            hgrid, wgrid = backbone.grid_size
            hpatch = conf.input_height // hgrid
            wpatch = conf.input_width // wgrid
            backbone.patch_size = (hpatch, wpatch)
            backbone.conv1.kernel_size = (hpatch, wpatch)
            backbone.conv1.stride = (hpatch, wpatch)
            logger.info(f'change input image_size from 224 to {conf.input_width}')

    elif backbone_name in ['convnext_xxlarge']:
        model, _, _ = open_clip.create_model_and_transforms(backbone_name, pretrained=conf.backbone_pretrained)
        backbone = model.visual

        if conf.set_grad_checkpointing:
            backbone.set_grad_checkpointing(True)
            logger.info('use Gradient Checkpointing')

        n_features = 1024

    elif backbone_name in ['convnext_large_d_320']:
        model, _, _ = open_clip.create_model_and_transforms(backbone_name, pretrained=conf.backbone_pretrained)
        backbone = model.visual

        if conf.set_grad_checkpointing:
            backbone.set_grad_checkpointing(True)
            logger.info('use Gradient Checkpointing')

        n_features = 768

    elif backbone_name in ['eva02_large_patch14_448.mim_m38m_ft_in22k']:
        backbone = timm.create_model(backbone_name, img_size=(conf.input_height, conf.input_width), pretrained=True,
                                     drop_path_rate=conf.eva_drop_path_rate)
        n_features = backbone.num_features
        backbone.reset_classifier(0)

        # headを消す
        backbone.head = nn.Identity()
        backbone.fc_norm = nn.Identity()

        # gradient checkpointingを有効にすると現状学習が落ちる？
        if conf.set_grad_checkpointing:
            backbone.set_grad_checkpointing()
            logger.info('use Gradient Checkpointing')

    else:
        raise NotImplementedError(f'not implemented yet: {backbone_name}')

    model_fn = model_dict[conf.model_name]
    model = model_fn(conf, backbone, n_features)

    if conf.ckpt_path:
        ckpt = torch.load(conf.ckpt_path, map_location='cpu')
        check = model.load_state_dict(ckpt['model'], strict=conf.load_strict)
        logger.info(f"load model weight from : {conf.ckpt_path}")
        logger.info(f"check: {check}")

    return model
