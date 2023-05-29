import os
from PIL import Image
from torch.utils.data import Dataset


class StableDiffusion2Dataset(Dataset):
    def __init__(self, dataset, vis_root):
        self.dataset = dataset
        self.vis_root = vis_root

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        image_path = self.dataset.loc[idx].image
        image_path = os.path.join(self.vis_root, image_path)
        prompt = self.dataset.loc[idx].prompt
        img = Image.open(image_path).convert('RGB')
        return {
            'image': img,
            'prompt': prompt,
        }


class ImageEmbeddingDataset(Dataset):
    def __init__(self, dataset, transform, feats):
        self.dataset = dataset
        self.transform = transform
        self.feats = feats
        assert len(dataset) == self.feats.shape[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        embedding = self.feats[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, embedding, idx


def get_dataset(
    conf,
    mode,
    dataframe,
    feats,
    vis_root,
):
    if conf.task_type == 'image_embedding':
        dataset = StableDiffusion2Dataset(dataframe, vis_root=vis_root)
        if mode == 'train':
            transform = get_transform(conf, mode=conf.train_trans_mode)
        elif mode == 'valid':
            transform = get_transform(conf, mode=conf.valid_trans_mode)
        dataset = ImageEmbeddingDataset(dataset, transform, feats)
    else:
        raise NotImplementedError(conf.task_type)

    return dataset


import torchvision.transforms as T
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, \
     ToTensor, Resize, ColorJitter


def get_transform(conf, mode='justresize'):
    interpolation_dict = {
        'BILINEAR': T.InterpolationMode.BILINEAR,
        'BICUBIC': T.InterpolationMode.BICUBIC,
        'LANCZOS': T.InterpolationMode.LANCZOS,
    }
    interpolation = interpolation_dict[conf.interpolation]

    interpolation = T.InterpolationMode.BICUBIC

    mean = [float(mean) for mean in conf.rgb_mean.split(',')]
    std = [float(std) for std in conf.rgb_std.split(',')]
    normalize = Normalize(mean=mean, std=std)

    if mode == 'justresize':
        transform = Compose([
            Resize((conf.input_height, conf.input_width), interpolation=interpolation, max_size=None, antialias=None),
            ToTensor(),
            normalize
        ])
    elif mode == 'aug_v1':
        transform = Compose([
            RandomResizedCrop(
                size=(conf.input_height, conf.input_width),
                scale=[conf.crop_scale_min,conf.crop_scale_max],
                ratio=[conf.crop_ratio_min,conf.crop_ratio_max],
                interpolation=interpolation
            ),
            ColorJitter(
                brightness=conf.jitter_brightness,
                contrast=conf.jitter_contrast,
                saturation=conf.jitter_saturation,
                hue=conf.jitter_hue
            ),
            ToTensor(),
            normalize
        ])
    else:
        raise NotImplemented(mode)

    return transform

