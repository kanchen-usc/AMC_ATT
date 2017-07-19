# AMC: Attention guided Multi-modal Correlation Learning for Image Search
This repository includes annotated keyword datasets used by AMC system (CVPR 2017)

## Introduction

**AMC System** is initially described in an [arxiv tech report](https://arxiv.org/abs/1704.00763). We leverage visual and textual modalities for image search by learning their correlation with input query. According to the intent of query, attention mechanism can be introduced to adaptively balance the importance of different modalities.

## Framework

The framework of AMC is shown below

<img src='img/pipeline.png' width='900'>

We propose a novel Attention guided Multi-modal Correlation (AMC) learning method which consists of a jointly learned hierarchy of intra and inter-attention networks. Conditioned on query's intent, intra-attention networks (i.e., visual intra-attention network and language intra-attention network) attend on informative parts within each modality; a multi-modal inter-attention network promotes the importance of the most query-relevant modalities.

## Keyword Dataset

To validate the effectiveness of AMC System, we annotated a keyword dataset for [Microsoft Clickture dataset](https://www.microsoft.com/en-us/research/project/clickture/) and [MSCOCO Image Caption dataset](http://mscoco.org/dataset/#captions-challenge2015) using an auto-tagging system for each image. Some of the results are shown below (left and right columns are selected samples from Clickture and MSCOCO caption dataset respectively.)

<p align="center">
  <img src='img/dataset.png' width='450'/>
</p>

The download link for keyword datasets is provided [here](https://www.dropbox.com/sh/g5fojtxuzpw8wo1/AAB9uFZsMsHpBLqw4E9ldgBda?dl=0) (DropBox). 
If you find our dataset useful in your research, please consider citing:
```
@inproceedings{chen2017amc,
  title={AMC: Attention guided Multi-modal Correlation Learning for Image Search},
  author={Chen, Kan and Bui, Trung and Chen, Fang and Wang, Zhaowen and Nevatia, Ram},
  booktitle={CVPR},
  year={2017}
}
```

## Keyword preprocessing

We offer python scripts to process original keywords to word IDs for each training and testing sample.

## Results visualization

The ROC curve and attention map visualization for Clickture dataset is shown below

<img src='img/vis_res.png' width='900'>

More details of the experiments are provided in our [paper](https://arxiv.org/abs/1704.00763).

## License

MIT