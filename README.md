# BC-Transformer
- Implementation of the paper "Imitation Game is Not Optimal: Alleviating Autoregressive Bias in Non-Autoregressive Transformers"
- In this paper, we propose BC-Transformrer, which extends the DA-Transformer in a bidirectional architecture.
<img src=BC-Arch.jpg width="1200px">
- We also propose the BCKD, which uses two symmetrical Autoregressive Transformers as the teacher model to generate the KD dataset.<div align=center><img src=BCKD-2.0.png width="500px"><div align=left>
- This repository contains the implementation of BC-Transformer, pretrained checkpoints, and the BCKD dataset.

# Codes of BC-Transformer
- The code of BC-Transformer extends from DA-Transformer v1.0.
- Additional Codes are placed in DA-Transformer/fs_plugins.
- exp-scripts contains the scripts of preprocess/training/evaluation/generation of WMT14 De-En Dataset.
# BCKD Datasets
- We release the BCKD dataset of WMT14 En-De/De-En, WMT16 En-Ro/Ro-En on the Google Cloud.
- Refer to the following link for BCKD data downloading:
   https://drive.google.com/drive/folders/1z3aw0ZiFTmpTWP8cc4mgnRcKCVKvQhdz?usp=sharing
