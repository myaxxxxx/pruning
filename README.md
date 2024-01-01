



<h2 align="center">
Efficient Speech-to-Text Translation: Progressive Pruning for Accelerated Speech Pre-trained Model
</h2>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/EMNLP-2023-brightgreen"> -->
  <!-- <under review><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a> -->
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>

<p align="center">
Codes for our regular and share adapters for speech-to-text translation tasks (Under review).
After the review period, we will open-source the code on our GitHub.
</p>

### Overview

<div style="text-align: center">
<img src="images/fig1.png"/>
</div>
<!-- ![](images/fig1.png#id=UEGkS&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) -->

### Installations

1. Create a conda environment with Pytorch:

```
conda create --name adapters python=3.9
conda activate adapters
```

2. Install fairseq

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# Next: Important!
python setup.py build develop
```

3. Other operations

    *Notes: Due to the version compatibility of packages, you also need to reinstall the following packages：*

```bash
# numpy np.float error 
pip install numpy==1.23.5

# generation error: sacrebleu import error TOKENIZER 
pip install sacrebleu==1.5.1
```

This repository is constructed using the codebase from fairseq. If you require information on the basic usage of fairseq, please refer to the [fairseq documentation](https://fairseq.readthedocs.io/en/latest/).

4. Other requirements

- pandas==2.0.3
- sacrebleu==1.5.1
- scikit-learn==1.3.0
- scipy==1.11.1
- sentencepiece==0.1.99
- tensorboard==2.14.0
- torch==2.0.1
- torchaudio==2.0.2
- tqdm==4.65.0




### Datasets and Models


<!-- #### Mustc v1 -->
#### Mustc Datasets Prepare

1. Please Download [Mustc-v1](https://docs.google.com/forms/d/e/1FAIpQLSer9jNfUtxbi610n3T6diXRlANBbuzShsCje-GtKs1Sngh0YQ/viewform?pli=1) datasets. 

   *Notes: It appears that the original dataset [website](https://www.fbk.eu/en/research-centers/) hides the download link. However the dataset can still be downloaded after filling out the dataset request [form](https://docs.google.com/forms/d/e/1FAIpQLSer9jNfUtxbi610n3T6diXRlANBbuzShsCje-GtKs1Sngh0YQ/viewform?pli=1) directly. So we recommend that you use this method.*

2. Make directories to store ST (MuST-C) and datasets. Please specify the target language.

```
TARGET=de
MUSTC_ROOT=data/mustc
```

2. Unzip the mustc datasets.
```
cd $MUSTC_ROOT
tar -xzvf MUSTC_v1.0_en-${TARGET}.tar.gz
```

#### Deltalm Prepare
1.  Download [Vocabulary](https://deltalm.blob.core.windows.net/deltalm/dict.txt), [ Sentencepiece-model](https://deltalm.blob.core.windows.net/deltalm/spm.model) and [Model](https://deltalm.blob.core.windows.net/deltalm/deltalm-base.pt) of deltalm and you need to tokenize raw data to spm data. 

2.  Preprocess spm data. 

#### Speech Pre-trained Model 

1. We use Hubert model for speech pre-trained model for training. Before training, please download the [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model.
