# Reading Time Modeling with Pixel-Based Language Models

This repository contains code for modeling reading time with GPT-2, mGPT, BERT, and pixel-based language models.

## Setup

### 1. Download the data

Download `data.tar.gz` from the [Dropbox folder](https://www.dropbox.com/scl/fo/wf2s4j5zw0z664pltvlyl/AOFUdXVoNh6WAXwPUyooVUo?rlkey=7zesvc72v7dreftm56soq4mtc&st=4wecdb6s&dl=0) and place it in the repository root, alongside `probe_vl/` and `scripts/`.

Extract the archive:

```bash
tar -xvzf data.tar.gz
```

### 2. Install the environment

Install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

For the exact original Conda environment, refer to `pixel_m4.yaml`. Alternatively, a prepacked `pixel-m4.tar.gz` environment is available in the same [Dropbox folder](https://www.dropbox.com/scl/fo/wf2s4j5zw0z664pltvlyl/AOFUdXVoNh6WAXwPUyooVUo?rlkey=7zesvc72v7dreftm56soq4mtc&st=4wecdb6s&dl=0).

### 3. Configure model checkpoints

Set the paths to the pixel-language-model checkpoints in `probe_vl/experiment_config.py`:

```python
DATASET_PARAS = '../scripts/probe_paras.yml'
PIXEL_M4 = "..."       # Base Pixel-M4 checkpoint (English and Chinese)
DE_PIXEL = "..."       # German checkpoint
PERSIAN_PIXEL = "..."  # Persian checkpoint
KO_PIXEL = "..."       # Korean checkpoint
ES_PIXEL = "..."       # Spanish checkpoint
TR_PIXEL = "..."       # Turkish checkpoint
```

## Running the probes

Run the GPT-2, mGPT, BERT, and Pixel-LM probes with:

```bash
cd probe_vl/
bash run.sh
```
