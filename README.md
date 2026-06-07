# CBraMod — Paper Investigation

[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)

This repository investigates **CBraMod**, a foundation model for EEG signals based 
on a criss-cross transformer architecture. It provides a short summary of the paper, 
reproduces its experiments, and explores potential improvements.

- **Notebook 1** — Reproduces the fine-tuning experiments from the paper on the SHU-MI 
motor imagery dataset.
- **Notebook 2** — Investigates strategies to enhance fine-tuning performance 
(gradual unfreezing, differential learning rates, dropout tuning).

---

## What problem does it address?

EEG signals present two core challenges :

1. **Spatio-temporal dependencies.** EEG carries both spatial information and temporal dynamics. 
Standard transformers, designed for language, treat all sequences equally, but in EEG, dependencies between patches cross-channel are essential. I.E. to syncronise for a movement a lot have to be evolved at the same time.

2. **Variability across datasets.** EEG recordings vary in the number of electrodes and signal duration depending on the acquisition setup and 
clinical context. A generalizable model must handle this heterogeneity.

---

## What they propose

### 1. Asymmetric Conditional Positional Encoding (ACPE)

Depthwise convolution layer that learns positional information dynamically from the input, rather than relying on fixed sinusoidal encodings as in the language models or learned parameter per channels as in LaBraM. The asymmetric design uses a large kernel to capture long-range spatial dependencies across EEG channels, and a smaller kernel for short-range temporal relationships across patches.

<div align="center">
<img src="figure/acpe.png" style="width: 50%;" />
</div>

### 2. Criss-Cross Transformer Layers

A custom attention mechanism that splits the attention heads into two groups: one 
attending along the **spatial axis** and one along the 
**temporal axis**.

<div align="center">
<img src="figure/criss-cross-attention.png" style="width: 50%;" />
</div>

---

## Experiments

A strong point of the paper is the number of experiments: the model is benchmarked 
across many datasets with varying numbers of channels and signal lengths, covering 
tasks from motor imagery to pathology detection. In my opinion these are the most significant results :

<div align="center">
<img src="figure/experiment-att.png" style="width: 50%;" />
</div>

<div align="center">
<img src="figure/archi-comparison-pretraining.png" style="width: 50%;" />
</div>

---

## Limitations

### 1. The model does not truly adapt to variable channel counts or signal lengths

Despite being presented as a generalizable foundation model, CBraMod handles 
dataset heterogeneity by fixing the input format: all signals are segmented into 
1-second windows, and the kernel size for the number of channels is kept constant across datasets. 

This means the positional encoding, which encodes a fixed `(n_channels, n_patches)` 
grid, is never truly transferable across datasets with different topologies. The 
paper does show that the ACPE helps during pretraining (see figure below), which 
makes sense: it provides a useful inductive bias. But it does not constitute genuine cross-dataset spatial generalization.

<div align="center">
<img src="figure/experiment-pe.png" style="width: 50%;" />
</div>

### 2. Fine-tuning is severely limited by the positional encoding design

Because the positional encoding is tied to a fixed spatial grid, transferring to a 
new dataset with a different electrode configuration requires re-initializing or 
re-learning the encoding from scratch. This largely undermines the benefit of 
pretraining on that component, and forces a full retraining cycle for each new 
dataset rather than lightweight fine-tuning. And makes finetuning essential, no zero shot capabilities which doesn't make sense for an LLM. It has a lot of parameters but is uncapable of zero-shot.

And it is shown in Notebook 1, where 

### Possible ameliorations 
1 . MAMBA : répond pas a une grande tache des modeles eeg qui est la prediction...
2 . Est-ce que la tache de fine-tuning est vraiment la bonne ? Talk about BrainJEPA
3 . Quand meme une grosse phase de fine tuning ensuite, on comprend que le pre training aide beaucoup mais tout de meme 
4 . Better in emotion recognition ? Why ?

A more flexible design such as coordinate-based encodings using 3D scalp geometry, 
or relative positional encodings (e.g. RoPE-style) — could enable genuine 
cross-dataset transfer without discarding the spatial knowledge acquired during 
pretraining.

