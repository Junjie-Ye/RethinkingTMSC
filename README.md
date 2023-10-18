# RethinkingTMSC

## RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification
> Dataset and codes for paper "RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification"

Junjie Ye

jjye23@m.fudan.edu.cn

Oct. 12, 2023

## Requirement
* Python 3.7+

- Run the command to install the packages required.
    ```bash
    pip install -r requirements.txt
    ```


## Download tweet images and ResNet-152
- Step 1: Download each [tweet's associated image](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view).
- Step 2: Save the images to `data/Twitter15/images/` and `data/Twitter17/images/`, respectively.
- Step 3: Download the pre-trained [ResNet-152](https://download.pytorch.org/models/resnet152-b121ed2d.pth).
- Setp 4: Put the pre-trained ResNet-152 model under the folder named `Code/resnet/`.

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
    cd scripts
    bash run.sh
    ```

### Reminder
- You can find the results we report in our paper from the `output/` folder directly.

## Acknowledgements

- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.

## Cite

- If you find our code is helpful, please cite our paper
```bibtex
@misc{ye2023rethinkingtmsc,
      title={RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification}, 
      author={Junjie Ye and Jie Zhou and Junfeng Tian and Rui Wang and Qi Zhang and Tao Gui and Xuanjing Huang},
      year={2023},
      eprint={2310.09596},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
