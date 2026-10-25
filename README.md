# Reading Time Modeling with Pixel-Based Language Models
<p align="center">
  <img src="assets/output.gif" alt="Demo" width="600">
  <img src="assets/output2.gif" alt="Demo" width="600">
</p>
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
Download the checkpoints from [Dropbox folder](https://www.dropbox.com/scl/fo/wf2s4j5zw0z664pltvlyl/AOFUdXVoNh6WAXwPUyooVUo?rlkey=7zesvc72v7dreftm56soq4mtc&st=4wecdb6s&dl=0). 
You can unzip it somewhere. Then set the paths to the pixel-language-model checkpoints in `probe_vl/experiment_config.py`:

```python
PIXEL_M4 = "..."       # Base Pixel-M4 checkpoint (English and Chinese)
DE_PIXEL = "..."       # German checkpoint
PERSIAN_PIXEL = "..."  # Persian checkpoint
KO_PIXEL = "..."       # Korean checkpoint
ES_PIXEL = "..."       # Spanish checkpoint
TR_PIXEL = "..."       # Turkish checkpoint
```

For example, you can
```bash
mkdir checkpoints
mv checkpoints.zip ./checkpoints/
cd checkpoints
unzip checkpoints.zip
```

Then the configuration should be

```python
PIXEL_M4 = "../checkpoints/PIXEL_M4"       # Base Pixel-M4 checkpoint (English and Chinese)
DE_PIXEL = "../checkpoints/DE_PIXEL"       # German checkpoint
PERSIAN_PIXEL = "../checkpoints/PERSIAN_PIXEL"  # Persian checkpoint
KO_PIXEL = "../checkpoints/KO_PIXEL"       # Korean checkpoint
ES_PIXEL = "../checkpoints/ES_PIXEL"       # Spanish checkpoint
TR_PIXEL = "../checkpoints/TR_PIXEL"       # Turkish checkpoint
```
## Running the probes

Run the GPT-2, mGPT, BERT, and Pixel-LM probes with:

```bash
cd probe_vl/
bash run.sh
```

The surprisal and entropy results are in probe_vl/precomputed


### 4. Analysis

Coming soon.