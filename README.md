# Pneumonia-DL-CAD

A Python-based deep learning approach to automated pneumonia classification, localisation, and report generation from chest X-rays (CXRs). Developed as part of a final-year undergraduate thesis at Imperial College London.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1a1c53b6-ba34-4fcf-8efa-4630b38095f1" alt="Right mid/lower zone pneumonia localisation" width=900/>
</p>
<p align="center"><em>Right mid/lower zone pneumonia localisation</em></p>

## Scripts
- `background_process.py` : Defines a `CXR_Processor` class for preprocessing CXRs. This includes cropping around the trunk region, enhancing contrast, removing background noise, and exporting both the processed images and a log of any failed processing attempts. Supports multiple image formats (e.g., JPG, PNG, DICOM).
  
- `background_labeller.py` : Uses a locally deployed LLM (`DeepSeek-R1-Distill-Llama-8B`) to analyse MIMIC-CXR radiology reports, labelling pneumonia status (positive, negative, or uncertain) and extracting positional information. Requires a Hugging Face token and assumes GPU availability.
  
- `background_train.py` : Trains a convolutional neural network (`DenseNet-121`) to classify pneumonia status from CXRs using the MIMIC-CXR and VinDr-CXR datasets. Incorporates data augmentation, transfer learning, early stopping, and checkpointing based on validation recall.

## Usage
These scripts are intended to run in the background due to the high volume of CXRs. For Unix systems:

```bash
nohup python3 -u /path/to/background_process.py > process.log 2>&1 &
```

To monitor progress:

```bash
tail -f process.log
```

## Notebooks

## Citation
