# RethinkingTMSC

## RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification
> Dataset and codes for paper "RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification"

Junjie Ye

jjye23@m.fudan.edu.cn

Oct. 12, 2023

## Requirement
* Python 3.7+


## Download tweet images and ResNet-152
- Step 1: Download each tweet's associated image via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view).
- Step 2: Save the images to `data/Twitter15/images` and `data/Twitter17/images`, respectively.
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth).
- Setp 4: Put the pre-trained ResNet-152 model under the folder named `Code/resnet`.

## Prepare image features
> Since the files are too big to load, please extract them by yourself.

- Step 1: Download the pre-trained [Pretrained Faster R-CNN model](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view?usp=sharing), which is trained with Visual Genome + Res101 + Pytorch and save it to the folder `Code/faster_rcnn/models/`.

- Step 2: Compile the cuda dependencies using following simple commands:
    ```bash
    cd Code/faster_rcnn
    python setup.py build develop
    ```

- Step 2: Extract the features and save them:
    ```bash
    cd Code/faster_rcnn
    python data_process.py --source_path ../../data/Twitter15/images --save_path ../../data/Twitter15/faster_features
    python data_process.py --source_path ../../data/Twitter17/images --save_path ../../data/Twitter17/faster_features
    ```


## Code Usage

### Training and Analysis
- This is the training code of tuning parameters on the dev set, and testing on the test set for all models.

    ```sh
    pip install -r requirements.txt
    cd scripts
    bash run.sh
    ```

### Reminder
- The file in `data/Twitter15/faster_features` and `data/Twitter15/faster_features` are the features extracted by Faster R-CNN ([Ren et al., 2015](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)), and you can also extract them by yourself.

- You can find the results we report in our paper from the `output/` folder directly.

## Acknowledgements

- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
