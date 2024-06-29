# BIOSCAN-CLIP
This is the official implementation for "BIOSCAN-CLIP: Bridging Vision and Genomics for Biodiversity Monitoring at Scale".

Links: [website](https://3dlg-hcvc.github.io/bioscan-clip/) | [paper](https://arxiv.org/abs/2405.17537)

# Overview
![Teaser](./docs/static/images/method.png)
Taxonomically classifying organisms at scale is crucial for monitoring biodiversity, understanding ecosystems, and preserving sustainability.  It is possible to taxonomically classify organisms based on their image or their [DNA barcode](https://en.wikipedia.org/wiki/DNA_barcoding).  While DNA barcodes are precise at species identification, they are less readily available than images.  Thus, we investigate whether we can use DNA barcodes to improve taxonomic classification using image.  

We introduce BIOSCAN-CLIP, a model uses contrastive learning to map biological images, textual taxonomic labels, and DNA barcodes to the same latent space.  The aligned image-DNA embedding space improves taxonomic classification using images and allows us to do cross-modal retrieval from image to DNA. We train BIOSCAN-CLIP on the [BIOSCAN-1M]([[https://www.kaggle.com/datasets/zahragharaee/bioscan-1m-insect-dataset](https://github.com/zahrag/BIOSCAN-1M)](https://biodiversitygenomics.net/projects/1m-insects/)) and [BIOSCAN-5M](https://biodiversitygenomics.net/projects/5m-insects/) insect datasets.  These datasets provides paired images of insects and their DNA barcodes, along with their taxonomic labels.  

# Setup environment
BIOSCAN-CLIP was developed using Python 3.10 and PyTorch 2.0.1.  We recommend the use of GPU and CUDA for efficient training and inference.  Our models were developed with CUDA 12.4.  
We also recommend the use of [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) for managing your environments. 

To setup the environment width necessary dependencies, type the following commands:
```shell
conda create -n bioscan-clip python=3.10
conda activate bioscan-clip
conda install pytorch=2.0.1 torchvision=0.15.2 torchtext=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
#pip install git+https://github.com/openai/CLIP.git
conda install -c conda-forge faiss
pip install .
```

Depending on your GPU version, you may have to modify the torch version and other package versions in [requirements.txt](https://github.com/3dlg-hcvc/bioscan-clip/blob/main/requirements.txt).

# Pretrained embeddings and models
We provide pretrained embeddings and models.  We evaluate our models by encoding the image or DNA barcode, and using the taxonomic labels from the closest matching embedding (either using image or DNA barcode).

| Training data |  Aligned modalities |  Embeddings |  Model  |  
|---------------|---------------------|-------------|---------|
| BIOSCAN-1M    |  None               |  TODO       |  TODO   | 
| BIOSCAN-1M    |  Image + DNA        |  TODO       |  TODO   | 
| BIOSCAN-1M    |  Image + DNA + Tax  |  TODO       |  TODO   | 
| BIOSCAN-5M    |  None               |  TODO       |  TODO   | 
| BIOSCAN-5M    |  Image + DNA        |  TODO       |  TODO   | 
| BIOSCAN-5M    |  Image + DNA + Tax  |  TODO       |  TODO   |

## Using pretrained models to extract embeddings

```shell
# From project folder
python scripts/extract_embedding.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
python scripts/extract_embedding.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```

# Download dataset
![Data Partioning Visual](./docs/static/images/partition.png) <br>
For BIOSCAN 1M, we partition the dataset for our BIOSCAN-CLIP experiments into a training set for contrastive learning, and validation and test partitions. The training set has records without any species labels as well as a set of seen species. The validation and test sets include seen and unseen species. These images are further split into subpartitions of queries and keys for evaluation.

For BIOSCAN 5M, we use the dataset partioning established in the BIOSCAN-5M paper.

For training and reproducing our experiments, we provide HDF5 files with BIOSCAN-1M and BIOSCAN-5M images.  We also provide scripts for generating the HDF5 files directly from the BIOSCAN-1M and BIOSCAN-5M data.

### Download BIOSCAN-1M data (79.7 GB)
TODO: add explanation of the hdf5 file.
```shell
# From project folder
mkdir -p data/BioScan_1M/split_data
cd data/BioScan_1M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/data/version_0.2.1/BioScan_data_in_splits.hdf5
```

### Download BIOSCAN-5M data (190.4 GB)
TODO: add the command for downloading the images and generating the hdf5 file.
```shell
# From project folder
mkdir -p data/BOSCAN_5M/split_data
cd data/BOSCAN_5M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_5M_for_downloading/BIOSCAN_5M.hdf5
```
# To download other data for generating hdf5 files.
You can check [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M) and [BIOSCAN-5M](https://github.com/zahrag/BIOSCAN-5M) to download tsv files.

# Running experiments
We recommend the use of [weights and biases](https://wandb.ai/site) to track and log experiments

## Activate Wandb
#### Register/Login for a [free wandb account](https://wandb.ai/site)
```shell
wandb login
# Paste your wandb's API key
```

## Checkpoints

Download checkpoint for BarcodeBERT and bioscan_clip
```shell
# From project folder
mkdir -p ckpt/BarcodeBERT/5_mer
cd ckpt/BarcodeBERT/5_mer
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/BarcodeBERT/model_41.pth
cd ../../..
mkdir -p ckpt/bioscan_clip
cd ckpt/bioscan_clip
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/bioscan_clip/version_0_1_0/lora_vit_lora_bert_ssl_batch_size_400/best.pth
cd ..
mkdir bioscan_clip_5m
cd bioscan_clip_5m
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_5M_for_downloading/ckpt/best.pth
```

## Train

To train using BIOSCAN-1M:

```shell
# From project folder
python scripts/train_cl.py 'model_config={config_name}'
```
For example
```shell
# From project folder
python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
```
For multiple GPU, you may have to
```shell
NCCL_P2P_LEVEL=NVL python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
```

To train using BIOSCAN-5M:
```shell
python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```

## Eval
Evaluation is done by predicting the accuracy of both seen and unseen species to test model generalizability. 

To run evaluation for BIOSCAN-1M:
```shell
# From project folder
python scripts/inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
```

To run evaluation for BIOSCAN-5M:
```shell
python scripts/inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```


# Conclusion
BIOSCAN-CLIP combines insect images with DNA barcodes and taxonomic labels to improve taxonomic classification via contrastive learning. This method capitalizes on the practicality and low cost of image acquisition, promoting wider participation in global biodiversity tracking. Experimental results demonstrate that BIOSCAN-CLIP's shared embedding space is effective for retrieval tasks involving both known and unknown species, and adaptable for downstream applications like zero-shot learning. The model faces challenges with underrepresented and unseen species, highlighting areas for future enhancement.
