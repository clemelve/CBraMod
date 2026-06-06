# CeBraMod

### What challenges does it try to solve
1 . EEG is a complex signal that involves both temporal and spatial dependancies, transformer architectures were made for language so not well suited for this kind of data.
2 . EEG signals are of variable length and channels depending on the application, need of generalizable format.

### What they propose 
1 . ACPE positional embedding : to dynamically learn spatial and temporal relationships 
2 . Criss-Cross transformer layers : type of attention mechanism where split the number of heads between spatial and temporal

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

