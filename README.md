# MVHR

This repository is the implementation of MVHR:

Multi-layer information fusion based on graph convolutional network(lightGCN) and Multi-view Contrastive Learning herb recommendation

The basic environment is Ubuntu 18.04 so that torch-1.5.0+cu102 can fit our experiment.

The main aim of this experiment:

>change the parameter
>
>Ablation learning

# Required packages

python==3.7.9(there are some bugs when use python>3.7.9 including(3.7.13 3.8+)

torch==1.5.0+cu102

torch-geometric

    torch-cluster==2.0.5

    torch-sparse==0.6.5

numpy==1.18.1

pandas==1.0.1

sklearn==0.22.1

faiss-gpu==1.7.1(abandoned)
