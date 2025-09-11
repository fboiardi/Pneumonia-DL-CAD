# Pneumonia-DL-CAD

A Python-based deep learning approach to automated pneumonia classification, localisation, and report generation from chest X-rays (CXRs). Developed as part of a final-year undergraduate thesis at Imperial College London.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1a1c53b6-ba34-4fcf-8efa-4630b38095f1" alt="Right mid/lower zone pneumonia localisation" width=900/>
</p>
<p align="center"><em>Right mid/lower zone pneumonia localisation</em></p>

## Scripts
- `background_process.py` : Defines a `CXR_Processor` class for preprocessing CXRs. Includes cropping around the trunk region, enhancing contrast, removing background noise, and exporting both processed images and a log of any failed processing attempts. Supports multiple image formats (e.g., JPG, PNG, DICOM).
  
- `background_labeller.py` : Uses a locally deployed LLM (`DeepSeek-R1-Distill-Llama-8B`) to analyse MIMIC-CXR radiology reports, labelling pneumonia status (positive/negative/uncertain) and extracting positional information. Requires a Hugging Face token and assumes GPU availability.
  
- `background_train.py` : Trains a convolutional neural network (`DenseNet-121`) to classify pneumonia status from CXRs using the MIMIC-CXR and VinDr-CXR datasets. Incorporates data augmentation, transfer learning, early stopping, and checkpointing based on validation recall.

## Usage
These scripts are intended to run in the background due to the high volume of CXRs. For Unix systems:

```bash
nohup python3 -u background_process.py > process.log 2>&1 &
```

To monitor progress:

```bash
tail -f process.log
```

## Notebooks
- `data_prep.ipynb` : Exploratory notebook for reformatting MIMIC-CXR and VinDr-CXR data, defining helper functions for CXR image access, and preliminary analyses like CXR view classification using a pre-trained model (`ResNet-50`).

- `mimic_reports.ipynb` : Parses MIMIC-CXR JSON radiology reports into unified texts and merges them with image metadata. Filters pneumonia-relevant cases and defines an LLM prompt to classify pneumonia (positive/negative/uncertain) and extract localisation information.

- `explainability.ipynb` : Consolidates and reformats CXR datasets for model training. Loads the trained `DenseNet-121` classifier to produce Grad-CAM visualisations and compare activations to lung-zone ground truth derived from report text using `DeepSeek-R1-Distill-Llama-8B`.

- `model_eval.ipynb` : Compares distributions of predicted pneumonia probabilities stratified by LLM-derived ground truth labels. Evaluates model performance across different CXR datasets, and includes error analysis focused on false-negative report text.

- `report_snippets.ipynb` : Generates radiology report snippets from pneumonia localisation strings using a locally run LLM.

## Requirements
Dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Citation
