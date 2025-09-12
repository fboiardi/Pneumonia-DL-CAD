import pandas as pd, numpy as np
from torchvision import transforms
import torch
import torchvision.models as models
from PIL import Image, ImageFile
import cv2, pydicom

ImageFile.LOAD_TRUNCATED_IMAGES = True

get_vindr_img_path = lambda vindr_row:f'/Data/VinDR_Data/physionet.org/files/vindr-cxr/1.0.0/{vindr_row["Set"]}/{vindr_row["image_id"]}.dicom'

def classify_vindr_quality(vindr_row):
    image_path = get_vindr_img_path(vindr_row)
    
    dicom = pydicom.dcmread(image_path)
    image_np = dicom.pixel_array

    # grayscale
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        image_np = image_np.max() - image_np

    # windowing
    if 'WindowCenter' in dicom:
        window_center = float(dicom.WindowCenter)
        window_width = float(dicom.WindowWidth)

        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        image_np = np.clip(image_np, lower_bound, upper_bound)

    image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # scale to 8-bit
    image = Image.fromarray(image_np, mode='L').convert('RGB')
    
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet50(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == '__main__':
    # data
    vindr = pd.read_csv('/Data/VinDR_Data/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_merged.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device.type}\n')

    # view classifier
    resnet50 = models.resnet50()
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 2) # two logits: frontal, lateral
    resnet50.load_state_dict(torch.load('/Data/jacky_models/resnet50_frontal_vs_lateral.pth'))

    resnet50 = resnet50.to(device)
    resnet50.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    vindr['Pred_View'] = vindr.apply(classify_vindr_quality, axis=1)
    vindr['Pred_View'] = vindr['Pred_View'].apply(lambda val : 'Lateral' if val else 'Frontal')

    vindr.to_csv('/Data/VinDR_Data/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_merged_w_views.csv', index=False)
    print('\nData saved')