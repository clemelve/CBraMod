# CeBraMod

This repository investigates the paper [![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236). This page contains a short summary on the paper. "Notebook 1" reproduces the experiments from the paper on the "SHU-MI" dataset. "Notebook 2" investigates enhancing performances of the paper. 

### What challenges does it try to solve
1 . EEG is a complex signal that involves both temporal and spatial dependancies, transformer architectures were made for language so not well suited for this kind of data. Indeed dependencies among EEG patches within the same channel or time interval could be stronger than those among patches from different channels and time intervals.
2 . EEG signals are of variable length and channels depending on the application. Some acquisition may be shorter and use different acquition schemes. There is a need of a model that could generalize to different format.

### What they propose 

1 . Asymmetric Conditional Positional Encoding (ACPE) : use a depthwise convolution layer to dynamically learn long-range spatial dependencies and short-range temporal relationships.
<div align="center">
<img src="figure/acpe.png" style="width: 50%;" />
</div>

2 . Criss-Cross transformer layers : type of attention mechanism where split the number of heads between spatial and temporal.
<div align="center">
<img src="figure/criss-cross-attention.png" style="width: 50%;" />
</div>

### Experiments
Did a lot of experiments and in my opinion, pretty good one, well done : 
1 or 2 most convincing experiments

### Limitations 
1 . Preprocessing : coupe les fréquences trop grandes, have to find a way.
2 . Although Positional Encoding seems to work, highlight that it is not the most efficient ever. 

### Possible ameliorations 
1 . MAMBA : répond pas a une grande tache des modeles eeg qui est la prediction...
2 . Est-ce que la tache de fine-tuning est vraiment la bonne ? 
3 . Quand meme une grosse phase de fine tuning ensuite, on comprend que le pre training aide beaucoup mais tout de meme 
4 . Better in emotion recognition ? Why ?

