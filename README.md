# Buffalo

Official implementation of [CIKM 2024] [Buffalo: Biomedical Vision-Language Understanding with Cross-Modal Prototype and Federated Foundation Model Collaboration](https://dl.acm.org/doi/10.1145/3627673.3679627).

## How to run

### Installation

```bash
pip install -r requirements.txt
```

### Model

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf
```

### Data Preparation

#### IU X-Ray

Download IU X-Ray dataset from [here](https://openi.nlm.nih.gov/faq) and extract the images to `data/iu_xray/NLMCXR_png` and the report to `data/iu_xray/NLMCXR_reports`.

#### Slake

Download Slake dataset from [here](https://www.med-vqa.com/slake/). Extract the images to `data/Slake/imgs` and the QA to `data/Slake/`.

#### VQA-RAD

Download VQA-RAD dataset from [here](https://osf.io/89kps/). Extract the images to `data/VQA-RAD/VQA_RAD Image Folder` and the QA to `data/VQA-RAD/`.

### Data Preprocessing

```bash
python dataset/prepare.py # remind to change the model path in the script
```

### Train

```bash
python main_multifed.py # remind to change the model path in the config file
```

## Citation

If you use Buffalo in your research, please cite the following paper:

```
@inproceedings{yan2024buffalo,
    author = {Yan, Bingjie and Chen, Qian and Chen, Yiqiang and Jiang, Xinlong and Huang, Wuliang and Wang, Bingyu and Wang, Zhirui and Gao, Chenlong and Zhang, Teng},
    title = {Buffalo: Biomedical Vision-Language Understanding with Cross-Modal Prototype and Federated Foundation Model Collaboration},
    year = {2024},
    isbn = {9798400704369},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627673.3679627},
    doi = {10.1145/3627673.3679627},
    booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
    pages = {2775–2785},
    numpages = {11},
    keywords = {biomedical vision-language understanding, cross-modal prototype, federated learning, modal heterogeneity, multi-modal},
    location = {Boise, ID, USA},
    series = {CIKM '24}
}
```