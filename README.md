# RethinkingTMSC

## RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification
- Dataset and codes for paper "RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification"

Author

Junjie Ye

jjye23@m.fudan.edu.cn

Oct. 12, 2023

## Requirement
* Python 3.7


## Download tweet images and set up image path
- Step 1: Download each tweet's associated image via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view).
- Step 2: Save the images to `data/Twitter15/images` and `data/Twitter17/images`, respectively.
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth).
- Setp 4: Put the pre-trained ResNet-152 model under the folder named `Code/resnet`.



## Code Usage

### Training and Analysis
- This is the training code of tuning parameters on the dev set, and testing on the test set for all models.

```sh
pip install -r requirements.txt
cd scripts
bash run.sh
```

### Reminder
- You can find the results we report in our paper from the `output/` folder directly.

## Acknowledgements

- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
