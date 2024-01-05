# ADGCCA: Attentive Deep Canonical Correlation Analysis for Diagnosing Alzheimer’s Disease Using Multimodal Imaging Genetics
## Overview
In this paper, a new model called attentive deep canonical correlation analysis (ADCCA) is proposed for the diagnosis of Alzheimer’s disease using multimodal brain imaging genetics data. ADCCA combines the strengths of deep neural networks, attention mechanisms, and canonical correlation analysis to integrate and exploit the complementary information from multiple data modalities. This leads to improved interpretability and strong multimodal feature learning ability. The ADCCA model is evaluated using the ADNI database with three imaging modalities (VBM-MRI, FDG-PET, and AV45-PET) and genetic SNP data. The results indicate that this approach can achieve outstanding performance and identify meaningful biomarkers for Alzheimer’s disease diagnosis. 
## Publication
- Title: ADGCCA: Advanced Deep Canonical Correlation Analysis
- Conference: Medical Image Computing and Computer Assisted Intervention – MICCAI 2023
- Link to Paper: [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_64)

## Key Components
- `ADCCA.py`: Defines the ADGCCA model classes for multi-modal data analysis.
- `utils.py`: Includes utility functions for data preprocessing, performance metric calculations, and other necessary operations.
- `train.py`: Script for training the ADGCCA model, performing hyperparameter optimization, and evaluating its performance.
- `env.yaml`: Conda environment file to set up necessary dependencies for replicating the study.

## Setup Instructions
1. **Environment Setup**:
   - Ensure that Conda is installed on your system.
   - Create a new Conda environment using the `env.yaml` file:
     ```
     conda env create -f env.yaml
     ```
   - Activate the environment:
     ```
     conda activate ADGCCA
     ```

2. **Data Preparation**:
   - Prepare your dataset according to the format expected by the ADGCCA model. The data should be split into four distinct modalities as required by the model.

## Running the Code
1. **Training the Model**:
   - Execute the `train.py` script to start training the ADGCCA model:
     ```
     python train.py
     ```
   - The training process involves hyperparameter tuning and validation through cross-validation techniques.

2. **Hyperparameter Tuning**:
   - Adjust the `hyper_dict` in the `train.py` script to experiment with different model configurations and training settings.

3. **Evaluation**:
   - The script outputs performance metrics like Accuracy, F1 Score, AUC, and Matthews Correlation Coefficient (MCC) for the model.

## Customization
- You may need to modify the model structure in `ADCCA.py` or the data preprocessing methods in `utils.py` to fit the specific requirements of your dataset.

## Citation
If you find this implementation useful, please consider citing our paper. Here is the BibTeX entry for the paper:

```bibtex
@inproceedings{zhou2023attentive,
  title={Attentive Deep Canonical Correlation Analysis for Diagnosing Alzheimer’s Disease Using Multimodal Imaging Genetics},
  author={Zhou, Rong and Zhou, Houliang and Chen, Brian Y and Shen, Li and Zhang, Yu and He, Lifang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={681--691},
  year={2023},
  organization={Springer}
}
