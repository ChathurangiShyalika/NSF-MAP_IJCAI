# Repository for "NSF-MAP: Neurosymbolic Multimodal Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines" Paper
This repository contains derived datasets, implementation of methods experimented and introduced in the paper titled "NSF-MAP: Neurosymbolic Multimodal Fusion for Robust and Interpretable
Anomaly Prediction in Assembly Pipelines".

# 1. Data Preprocessing #

This folder contains the codes used to extract the analog and multimodal datasets which was created from the Future Factories(FF) Dataset (https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset)
[Codes for preprocessing:Multimodal dataset preprocessing.ipynb, Analog dataset preprocessing.ipynb, Combine_MM_Data.ipynb]

The final preprocessed multimodal dataset available at: https://drive.google.com/drive/folders/1l7_Blmk_RrsLHqcW_4F7-ELtll_ZGyDv?usp=sharing

The final preprocessed analog dataset available at: https://drive.google.com/drive/folders/1v-UyJqlZMG68Mwd2Gict9E885MS023iR?usp=sharing

It also includes the additional experiments done to investigate the feature importance of sensor variables.[Feature_importance_experiments.ipynb]

# 2. Baselines # 

This folder includes the baseline models developed.
Three baseline models:

## Autoencoder ##
To run py .Baselines/autoencoder.py

## Custom ViT ##
Reproduced from the method implemented in the paper "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines"
To run py .Baselines/custom_vit.py

## CNN ##
Reproduced from the method implemented in the paper "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines"
To run py .Baselines/image_with_seg_cnn.py

## Pretrained-CNN ##
Reproduced from the method implemented in the paper "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines"
To run py .Baselines/image_with_segmentation_pretrainedcnn.py

## Pretrained-ViT ##
Reproduced from the method implemented in the paper "AssemAI: Interpretable Image-Based Anomaly Prediction for Manufacturing Pipelines"
To run py .Baselines/image_with_segmentation_vit.py

# 3. Proposed Fusion Approach #

This folder includes the models for the proposed approach.
## Decision level fusion ##
To run py .Proposed Fusion Approach/1.DLF.py

## Decision level fusion with transfer learning ##
To run py .Proposed Fusion Approach/2.DLF_TL.py

## Enhanced Decision-Level Fusion with Transfer Learning through Neurosymbolic AI ##
For random data splitting---
To run py .Proposed Fusion Approach/3.DLF_TL_KI.py

For time based (cycle-wise) data splitting---
To run py .Proposed Fusion Approach/4.DLF_TL_KI_timebased.py

# 4. Future Factories Setup #
This folder includes sample images obtained from the Future Factories(FF) Setup at University of South Carolina, USA 


# 5. Demo
NSF-MAP and process ontology is deployed at the FF testbed at the McNair Center, University of South Carolina
This folder includes the inference codes, user interface code for deployment, and the demo of deployment.

Demo video is available at: https://shorturl.at/IY9tt


# 6. Additional Experiments on NSF-MAP 
This folder includes additional experimental results of NSF-MAP and ablation Studies. The document includes the Mean Squared Errors (MSE) and Root Mean Squared Errors (RMSE) for the
sensor values and predicted anomaly labels. The folder also includes additional experiments on the process ontology.