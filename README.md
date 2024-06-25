# BIOSCAN-CLIP
This is the official implementation for "BIOSCAN-CLIP: Bridging Vision and Genomics for Biodiversity Monitoring at Scale".

Links: [website](https://3dlg-hcvc.github.io/bioscan-clip/) | [paper](https://arxiv.org/abs/2405.17537)

# Overview
![Teaser](./docs/static/images/method.png)
Taxonomically classifying organisms at scale and in-depth is crucial for monitoring biodiversity, understanding ecosystems, and preserving sustainability.
BIOSCAN-CLIP uses contrastive learning to map biological images, textual taxonomic labels, and DNA barcodes to the same latent space, while relaxing the constraint for comprehensive and correct taxonomy annotations. Either images or DNA barcodes can be flexibly classified to predict the taxonomy. The shared embedding space further enables future research into commonalities and differences between species. In addition to the ecological benefits, building such a foundation model for biodiversity is a case study of a broader challenge to build models which can identify fine-grained differences, both visually and textually. Taxonomic classification is particularly interesting because those visual differences between species are often not well-defined, and the DNA and text modalities, while using identical characters to those expected by most language models, do not share much semantic overlap with natural language. We demonstrate the benefits of pretraining with all three modalities through improved taxonomic classification accuracy over prior works in both retrieval and zero-shot settings using our learned representations.
# Dataset
BIOSCAN-CLIP uses the [BIOSCAN-1M dataset](https://www.kaggle.com/datasets/zahragharaee/bioscan-1m-insect-dataset
), a currated collection of over one million insect data records.
Insects comprise of vast biodiversity although approximately only 20% of them are described. They have essential applications to sectors such as agriculture making them an important sup-space to explore.
### i) Dataset Consists Of
- High quality insect images
- Expert- annotated taxonomic labels
- DNA barcodes
### ii) Data Partition
![Data Partioning Visual](./docs/static/images/partition.png) <br>
The data has been partitioned into a training set for contrastive learning, and validation and test partitions. The training set has records without any species labels as well as a set of seen species. The validation and test sets include seen and unseen species. These images are further split into subpartitions of queries and keys for evaluation.
<br>
# Experiments

Experiments using our model have assessed taxonomic classification accuracy via contrastive learning. Input images are matched against the closest DNA barcodes or comparable images for classification. 

# Set environment
For now, you can set the environment by typing
```shell
conda create -n bioscan-clip python=3.10
conda activate bioscan-clip
conda install pytorch=2.0.1 torchvision=0.15.2 torchtext=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
#pip install git+https://github.com/openai/CLIP.git
conda install -c conda-forge faiss
pip install .
```
in the terminal. However, based on your GPU version, you may have to modify the torch version and install other packages manually with different version.
# Download BIOSCAN-1M data
TODO: add some explaination of the hdf5 file.
```shell
# From project folder
mkdir -p data/BioScan_1M/split_data
cd data/BioScan_1M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/data/version_0.2.1/BioScan_data_in_splits.hdf5
```
# Download BIOSCAN-5M data
Note: add the command for downloading the images and generating the hdf5 file.
```shell
# From project folder
mkdir -p data/BOSCAN_5M/split_data
cd data/BOSCAN_5M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_5M_for_downloading/BIOSCAN_5M.hdf5
```
# Download checkpoint for BarcodeBERT and bioscan_clip
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
# Activate Wandb
#### Register/Login for a [free wandb account](https://wandb.ai/site)
```shell
wandb login
# Paste your wandb's API key
```
# Train
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
To train with 5M's data.
```shell
python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```
# Eval
Evaluation is done by predicting the accuracy of both seen and unseen species to test model generalizability. 
```shell
# From project folder
python scripts/inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
```
To eval with BIOSCAN-5M's data
```shell
python scripts/inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```
# Extract embedding
```shell
# From project folder
python scripts/extract_embedding.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
python scripts/extract_embedding.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_5m'
```

# Conclusion
BIOSCAN-CLIP combines insect images with DNA barcodes and taxonomic labels to improve taxonomic classification via contrastive learning. This method capitalizes on the practicality and low cost of image acquisition, promoting wider participation in global biodiversity tracking. Experimental results demonstrate that BIOSCAN-CLIP's shared embedding space is effective for retrieval tasks involving both known and unknown species, and adaptable for downstream applications like zero-shot learning. The model faces challenges with underrepresented and unseen species, highlighting areas for future enhancement.
