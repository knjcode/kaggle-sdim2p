# kaggle-sdim2p

5th place solution of [Stable Diffusion - Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts)

See [overview.md](overview.md) for an overview of our solution.

## Setup

### git clone and install packages

```
$ git clone https://github.com/knjcode/kaggle-sdim2p
$ cd kaggle-sdim2p
$ pip install -r requirements.txt
```

## Generate dataset

Please prepare the images yourself according to the following instructions, as we are not able to publish the image files we have generated.

Generate images from various prompts and save the image files under the `input` directory. In addition, create and save a tsv file that includes three items: image file path, prompt, and fold, in the following format:

```
image   prompt  fold
cc3m_images/08136.webp  a hot paper cup at coffee shop  0
cc3m_images/08137.webp  a hot paper cup at coffee shop  0
cc3m_images/08138.webp  a hot paper cup at coffee shop  0
cc3m_images/00898.webp  senior ceo sitting at meeting table     3
cc3m_images/00899.webp  senior ceo sitting at meeting table     3
cc3m_images/00900.webp  senior ceo sitting at meeting table     3
cc3m_images/22025.webp  zoot suit this style of clothing        5
cc3m_images/22026.webp  zoot suit this style of clothing        5
cc3m_images/22027.webp  zoot suit this style of clothing        5
cc3m_images/03893.webp  zucchini and italian dish is great      9
cc3m_images/03894.webp  zucchini and italian dish is great      9
cc3m_images/03895.webp  zucchini and italian dish is great      9
```

To partition the generated image-prompt pairs for training and evaluation, we first calculated the prompt embeddings using a sentence transformer. We then grouped samples with similar prompt embeddings. After grouping samples with a cosine similarity of 0.7 or higher in their embeddings, we divided the dataset into 10 folds based on these groups using GroupKFold. Model training was conducted on all folds except fold0, which was reserved for evaluation.

Furthermore, during the creation of the cc3m and diffusiondb datasets, we precomputed the cosine similarity of prompt embeddings. For prompts with a similarity of 0.9 or above, we only used one for image generation. From the midpoint of the competition, we also began to utilize the remaining prompts that we had not used before, creating the cc3m_part2 and diffusiondb_part2 datasets.

After creating tsv files, use the `input/gen_prompt_embedding.py` script to pre-calculate the sentence transformer embeddings from the prompts and save them to a file.

We generated approximately 5 million pairs of prompt-image sets (some images were generated multiple times from the same prompt) as detailed in the table below:

|dataset name     |# of prompts|images per prompt|Total images|
|:----------------|-----------:|----------------:|-----------:|
|cc3m             |     249,593|                3|     748,781|
|lexica           |      67,101|                3|     201,303|
|diffusiondb      |     326,035|                3|     978,105|
|cc3m part2       |   1,166,852|                1|   1,166,852|
|diffusiondb part2|   1,371,480|                1|   1,371,480|
|mscoco           |     197,527|                3|     592,583|
|Total:           |   3,378,588|             1or3|   5,059,104|

During training, for samples with three images corresponding to a prompt, we randomly selected one image to ensure that duplicate images from the same prompt were not used within the same epoch.

Dataset references
- cc3m: https://ai.google.com/research/ConceptualCaptions/download
- lexica: https://www.kaggle.com/datasets/safavieh/gustavosta-stable-diffusion-prompts
- diffusiondb: https://huggingface.co/datasets/poloclub/diffusiondb
- mscoco: https://cocodataset.org/


## Reproduce final submission

### Training models

We use A100(80GB)x8 hardware for training models.

#### Stage1 train models

move `kaggle-sdim2p/working` directory.

```bash
$ cd kaggle-sdim2p/working
```

Training stage1 models as follows:

```bash
## train convnext_large (4folds)
$ bash dist_train.sh 8 conf/convnext_large_3data_input384_bs32_lr1e-4_1e-6_wd5e-2_madgrad_wawrmup3_15epoch.yaml
... wait
## train vit_large (4folds)
$ bash dist_train.sh 8 conf/vit_large_3data_input336_bs32_lr1e-4_1e-6_wd5e-2_madgrad_wawrmup3_15epoch.yaml
... wait
## train convnext_xxlarge (4folds)
$ bash dist_train.sh 8 conf/convnext_xxlarge_3data_input384_bs32_lr1e-4_1e-6_wd5e-2_madgrad_wawrmup1_10epoch.yaml
... wait
## train eva02_large (2folds)
$ bash dist_train.sh 8 conf/eva02_large_clsavg_3data_input448_bs32_lr1e-4_1e-6_wd5e-2_madgrad_wawrmup1_10epoch.yaml
... wait
## train eval02_large (2folds)
$ bash dist_train.sh 8 conf/eva02_large_clsonly_3data_input448_bs32_lr1e-4_1e-6_wd5e-2_madgrad_wawrmup1_10epoch.yaml
... wait
```

#### Stage2 refine models

Training stage2 models as follows:

```bash
## refine convnext_large (4folds)
$ bash dist_train.sh 8 conf/convnext_large_8data_noaug_input384_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch.yaml
... wait
## refine vit_large (4folds)
$ bash dist_train.sh 8 conf/vit_large_8data_noaug_input336_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch.yaml
... wait
## refine convnext_xxlarge (4folds)
$ bash dist_train.sh 8 conf/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch.yaml
... wait
## refine eva02_large (2folds)
$ bash dist_train.sh 8 conf/eva02_large_clsavg_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch.yaml
... wait
## refine eva02_large (2folds)
$ bash dist_train.sh 8 conf/eva02_large_clsonly_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch.yaml
... wait

```

Refine plus 1 epoch for convnext_xxlarge and eva02_large as follows:

```bash
## refine plus 1 epoch convnext_xxlarge
$ bash dist_train.sh 8 conf/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_1epoch_fold6.yaml
... wait
## refine plus 1 epoch eva02_large
$ bash dist_train.sh 8 conf/eva02_large_clsavg_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_1epoch_fold2.yaml
... wait
```

You can obtain results equivalent to a final submission by using the model trained in this procedure and utilizing the notebook at the following URL:

[https://www.kaggle.com/code/knjcode/kydrt-5th-place-submission](https://www.kaggle.com/code/knjcode/kydrt-5th-place-submission)


For convnext_large use 4 models as follows (fold0,1,2,3):
```
models/convnext_large/convnext_large_8data_noaug_input384_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold0_best_cossim.pth
models/convnext_large/convnext_large_8data_noaug_input384_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold1_best_cossim.pth
models/convnext_large/convnext_large_8data_noaug_input384_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold2_best_cossim.pth
models/convnext_large/convnext_large_8data_noaug_input384_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold3_best_cossim.pth
```

For vit_large use 4 models as follows (fold0,3,4,8):
```
models/vit_large/vit_large_8data_noaug_input336_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold0_best_cossim.pth
models/vit_large/vit_large_8data_noaug_input336_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold3_best_cossim.pth
models/vit_large/vit_large_8data_noaug_input336_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold4_best_cossim.pth
models/vit_large/vit_large_8data_noaug_input336_bs16_lr1e-5_1e-7_wd5e-2_sam_madgrad_wawrmup0_5epoch_fold8_best_cossim.pth
```

For convnext_xxlarge use 5 models as follows (fold0,1,6,7,6*):
```
models/convnext_xxlarge/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold0_best_cossim.pth
models/convnext_xxlarge/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold1_best_cossim.pth
models/convnext_xxlarge/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold6_best_cossim.pth
models/convnext_xxlarge/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold7_best_cossim.pth
models/convnext_xxlarge/convnext_xxlarge_8data_noaug_input384_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_1epoch_fold6_best_cossim.pth
```

For eva02_large use 5 models as follows (fold0,2,5,9,2*):
```
models/eval02_large/eva02_large_clsavg_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold0_best_cossim.pth
models/eval02_large/eva02_large_clsavg_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold2_best_cossim.pth
models/eval02_large/eva02_large_clsonly_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold5_best_cossim.pth
models/eval02_large/eva02_large_clsonly_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_3epoch_fold9_best_cossim.pth
models/eval02_large/eva02_large_clsavg_8data_noaug_input448_bs16_lr1e-6_1e-7_wd5e-2_sam_madgrad_wawrmup0_1epoch_fold2_best_cossim.pth
```


## License

MIT
