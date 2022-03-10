# [Hierarchical Sketch Induction for Paraphrase Generation](https://arxiv.org/abs/2203.03463)


<img src="https://raw.githubusercontent.com/tomhosking/hrq-vae/main/hrqvae_gm.png" width="300" align="right" alt="Graphical Model diagram for HRQ-VAE" />



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-mscoco)](https://paperswithcode.com/sota/paraphrase-generation-on-mscoco?p=hierarchical-sketch-induction-for-paraphrase)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-paralex)](https://paperswithcode.com/sota/paraphrase-generation-on-paralex?p=hierarchical-sketch-induction-for-paraphrase)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-sketch-induction-for-paraphrase/paraphrase-generation-on-quora-question-pairs-1)](https://paperswithcode.com/sota/paraphrase-generation-on-quora-question-pairs-1?p=hierarchical-sketch-induction-for-paraphrase)

This repo contains the code for the paper [Hierarchical Sketch Induction for Paraphrase Generation](https://arxiv.org/abs/2203.03463), by Tom Hosking, Hao Tang & Mirella Lapata (ACL 2022).


## Installation

First, create a fresh virtualenv and install TorchSeq and other dependencies:
```
pip install -r requirements.txt
```

Then download (or create) the datasets/checkpoints you want to work with:

<a href="https://tomho.sk/models/separator/data_paralex.zip" download>Download our split of Paralex</a>

<a href="https://tomho.sk/models/separator/data_qqp.zip" download>Download our split of QQP</a>

<a href="https://tomho.sk/models/hrqvae/data_mscoco.zip" download>Download our split of MSCOCO</a>

<a href="https://tomho.sk/models/hrqvae/hrqvae_wa.zip" download>Download a pretrained checkpoint for Paralex</a>

<a href="https://tomho.sk/models/hrqvae/hrqvae_qqp.zip" download>Download a pretrained checkpoint for QQP</a>

<a href="https://tomho.sk/models/hrqvae/hrqvae_mscoco.zip" download>Download a pretrained checkpoint for MSCOCO</a>

Checkpoint zip files should be unzipped into `./models`, eg `./models/hrqvae_qqp`. Data zip files should be unzipped into `./data/`.

Note: Paralex was originally scraped from WikiAnswers, so many of the Paralex models and datasets are labelled as 'wa' or WikiAnswers.

## Inference with pre-trained checkpoints

To replicate our results (eg for QQP), have a look at the example in `./examples/Replication-QQP.ipynb`.

## Inference on a custom dataset

You can also run the model on your own data:

```
import json
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

import torch

# Which checkpoint should we load?
path_to_model = './models/hrqvae_paralex/'
path_to_data = './data/'

# Define the data
examples = [
    {'input': 'What is the income for a soccer player?'},
    {'input': 'What do soccer players earn?'}
]


# Change the config to use the custom dataset
with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["dataset"] = "json"
cfg_dict["json_dataset"] = {
    "path": None,
    "field_map": [
        {"type": "copy", "from": "input", "to": "s2"},
        {"type": "copy", "from": "input", "to": "s1"},
    ],
}

# Enable the code predictor
cfg_dict["bottleneck"]["code_predictor"]["infer_codes"] = True

# Create the dataset and model
config = Config(cfg_dict)
data_loader = JsonDataLoader(config, test_samples=examples, data_path=path_to_data)
checkpoint_path = path_to_model + "/model/checkpoint.pt"
instance = ParaphraseAgent(config=config, run_id=None, output_path=None, data_path=path_to_data, silent=True, verbose=False, training_mode=False)

# Load the checkpoint
instance.load_checkpoint(checkpoint_path)
instance.model.eval()
    
# Finally, run inference
_, _, (pred_output, _, _), _ = instance.inference(data_loader.test_loader)

print(pred_output)
```
> ['what is the salary for a soccer player?', 'what do soccer players earn?']

## Training from scratch

Train a fresh checkpoint using:

```
torchseq --train --config ./configs/hrqvae_paralex.json
```

## Training on a new dataset

To use a different dataset, you will need to generate a total of 4 datasets. These should be folders in `./data`, containing `{train,dev,test}.jsonl` files.

An example of this process is given in `./scripts/MSCOCO.ipynb`.

#### A cluster dataset, that is a list of the paraphrase clusters

```
{"qs": ["What are some good science documentaries?", "What is a good documentary on science?", "What is the best science documentary you have ever watched?", "Can you recommend some good documentaries in science?", "What the best science documentaries?"]}
{"qs": ["What do we use water for?", "Why do we, as human beings, use water for?"]}
...
```

#### A flattened dataset, that is just a list of all the paraphrases

The sentences must be in the same order as in the cluster dataset!

```
{"q": "Can you recommend some good documentaries in science?"}
{"q": "What the best science documentaries?"}
{"q": "What do we use water for?"}
...
```

#### The training triples

Generate this using the following command for question datasets:

```
python3 ./scripts/generate_training_triples.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 26 --extended_stopwords  --real_exemplars --exhaustive --template_dropout 0.3 --dataset qqp-clusters --min_samples 0
```

Or this command for other datasets:
```
python3 ./scripts/generate_training_triples.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 26 --pos_templates --extended_stopwords --no_stopwords  --real_exemplars --exhaustive --template_dropout 0.3 --dataset mscoco-clusters --min_samples 0
```

Replace `qqp-clusters` with the path to your dataset in "cluster" format.


#### A dataset to use for evaluation

For each cluster, select a single sentence to use as the input (assigned to `sem_input`) and add all the other references to `paras`. `tgt` and `syn_input` should be set to one of references.

```
{"tgt": "What are some good science documentaries?", "syn_input": "What are some good science documentaries?", "sem_input": "Can you recommend some good documentaries in science?", "paras": ["What are some good science documentaries?", "What the best science documentaries?", "What is the best science documentary you have ever watched?", "What is a good documentary on science?"]}
{"tgt": "What do we use water for?", "syn_input": "What do we use water for?", "sem_input": "Why do we, as human beings, use water for?", "paras": ["What do we use water for?"]}
...
```

#### Train the model

Have a look at the config files, eg `configs/hrqvae_qqp.json`, and update all the references to the different datasets, then run:

`torchseq --train --config ./configs/hrqvae_mydataset.json`

## Use HRQ-VAE in your project

Have a look at `./src/hrq_vae.py` for our implementation.


## Citation

```
@misc{hosking2022hierarchical,
    title={Hierarchical Sketch Induction for Paraphrase Generation},
    author={Tom Hosking and Hao Tang and Mirella Lapata},
    year={2022},
    eprint={2203.03463},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
