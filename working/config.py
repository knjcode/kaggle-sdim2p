import yaml
from dataclasses import dataclass


@dataclass
class Config:
    save_dir: str = 'vit_large'
    prefix: str = 'vit_large_finetune'

    task_type: str = 'image_embedding'  # image_embedding only

    n_fold: int = 10  # 現状10のみ
    train_fold: str = '0'  # fold0のみ学習  or '0,1,2,3,4,5,6,7,8,9'指定で全fold学習
    val_fold: int = None  # valするfoldを変更する場合に指定

    extract_feat: bool = False # 特徴抽出のみ実施

    base_dir: str = '../input/'
    dataset_name: str = 'cc3m_grouping'

    torch_compile: bool = False  # torch.compileする
    set_grad_checkpointing: bool = False  # Gradient Checkpointing

    seed: int = 71

    amp: bool = True
    amp_dtype: str = 'float16'         # float16 / bfloat16(A100のみ対応)

    model_name: str = 'ViTHeadlessModel'  # ViTHeadlessModel / ViTWithTokenModel

    # 同一promptのデータを1件に絞る(epoch毎に絞り直し)
    drop_duplicates: bool = True

    # for Image_embedding
    backbone_name: str = 'ViT-L-14'
    backbone_pretrained: str = 'laion2b_s32b_b82k'
    no_fc_bias: bool = True

    final_norm_layer: str = 'none'  # none / batch: nn.BatchNorm1d / layer: nn.LayerNorm
    # BatchNorm1d利用時は final_norm_no_decayをfalseにする
    final_norm_no_decay: bool = True  # final_norm_layerにweight_decayを適用しない
    final_act_layer: str = 'prelu'  # none / relu / swish(silu) / selu / prelu / mish / leaky_relu / gelu / sigmoid ...
    final_act_multi_dim: bool = False  #  PReLUの場合に入力次元を384にする
    final_act_inplace: bool = True

    backbone_freeze_epochs: int = 0  # 1で1epoch目のみfreeze

    # timmのbackboneのオプション
    eva_drop_path_rate: float = 0.         # timmのeva向けのdrop_path_rate
    # remove_last_stride: bool = False

    # pooling
    pooling: str = 'ggemp' # avg / max / gem / ggemp / avgmax/ catavgmax / nop

    # ViTModel
    h_patch: int = 12
    w_patch: int = 12
    with_cls: bool = False  # clsトークンの有無
    cls_mode: str = 'avg'  # avg / cat / cls_only / without_cls / full_avg

    # cls and/or tokens
    merge_mode: str = 'cat'  # tokens_only / avg / cat

    # GeM seggings
    gemp_p: float = 5.0
    gemp_learn_p: bool = True

    # GGeM settings
    ggemp_p: float = 5.0
    ggemp_learn_p: bool = True
    ggemp_num_groups: int = 12  # ViT-L=16

    # load pretraind model manually
    ckpt_path: str = ''
    load_strict: bool = True

    # image size
    input_width: int = 224
    input_height: int = 224

    # batch size
    train_bs: int = 64
    valid_bs: int = 64

    # optimizer settings
    optimizer: str = 'adamw'
    lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 5e-2
    momentum: float = 0.9
    nesterov: bool = False
    amsgrad: bool = False
    madgrad_decoupled_decay: bool = True

    # SAM
    sam: bool = False
    sam_adaptive: bool = False

    # scheduler settings
    epochs: int = 10
    warmup_epochs: int = 1
    warmup_lr_init: float = 1e-7
    plateau_mode: str = 'max'
    factor: float = 0.1
    patience: int = 4
    plateau_eps: float = 1e-8
    scheduler: str = 'cosine'  # ReduceLROnPlateau / CosineAnnealingLR /
                               # WarmupLinear / cosine
    cosine_interval_type: str = 'step'  # step or epoch  CosineLRSchedulerのstep頻度

    # augmentation
    train_trans_mode: str = 'aug_v1'  # aug_v1 / justresize
    valid_trans_mode: str = 'justresize'  # justresize

    interpolation: str = 'BICUBIC'  # BILINEAR / BICUBIC / LANCZOS

    # for torchvision.transforms
    rgb_mean: str = "0.48145466,0.4578275,0.40821073"
    rgb_std: str = "0.26862954,0.26130258,0.27577711"

    jitter_brightness: float = 0.05
    jitter_contrast: float = 0.05
    jitter_saturation: float = 0.05
    jitter_hue: float = 0.05

    crop_scale_min: float = 0.5
    crop_scale_max: float = 1.0
    crop_ratio_min: float = 0.9
    crop_ratio_max: float = 1.1

    train: bool = True
    save_cossim_list: bool = True
    save_prompts: bool = True
    save_feats: bool = False

    debug: bool = False  # データ数を減らす
    skip_val: bool = False   # valをskip
    save_every_epoch: bool = False  # 全epohcでモデルをsaveする

    num_workers: int = 4
    device: str = 'cuda'
    pin_memory: bool = True
    model_dir: str = 'models'
    log_dir: str = 'logs'
    print_freq: int = 0


def load_config(config_file):
    with open(config_file, 'r') as fp:
        opts = yaml.safe_load(fp)
    return Config(**opts)

