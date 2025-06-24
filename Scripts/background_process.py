import torch, torch.nn.functional as F, torchxrayvision as xrv
import numpy as np, pandas as pd, pydicom, cv2, os
from numpy.ma import masked_array
from scipy.ndimage import gaussian_filter1d, label
from scipy.signal import find_peaks
from skimage.exposure import histogram

class CXR_Processor:
    def __init__(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available
        print(f'Device: {self._device}\n')
        
        self._model = xrv.baseline_models.chestx_det.PSPNet().to(self._device) # pre-trained PSPNet model
        self._resizer = xrv.datasets.XRayResizer(512, engine='cv2') # resizes images to 512x512 pixels (ignore engine message)

        self.failed_cxrs = {} # failed processing attempts

        self.__count = 1 # keep track of cxrs processed

    @staticmethod
    def load_dicom(img_path:str):
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array

        # convert to standard grayscale
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img

        # windowing
        if 'WindowCenter' in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)

            lower_bound = window_center - (window_width / 2)
            upper_bound = window_center + (window_width / 2)
            img = np.clip(img, lower_bound, upper_bound)

        return img
    
    def load_and_process(self, img_path:str):
        """
        Loads and processes CXRs for segmentation.
        The PSPNet requires CXRs normalised to [-1024, 1024] and resized to 512x512 pixels as input
        * Note: converts images to 8-bit range

        PARAMETERS
        ----------
        - img_path (str) : path to CXR image

        RETURNS
        -------
        - img (numpy.ndarray) : original CXR
        - img_processed (torch.Tensor) : CXR processed for segmentation
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image path not found: {img_path}\n')
        
        img = self.load_dicom(img_path) if img_path.endswith(('.dcm', '.dicom')) else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img.dtype != 'uint8': # convert to 8-bit if necessary
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # normalise to [-1024, 1024] and add channel
        img_normalised = xrv.datasets.normalize(img, 255)[None, ...]

        # resize to 512x512
        img_processed = torch.from_numpy(self._resizer(img_normalised)).to(self._device)
    
        return img, img_processed

    @staticmethod
    def mask_bounding_box(mask, img_height, img_width, pad_fraction):
        """
        Calculates bounding box coordinates surounding composite trunk mask.

        PARAMETERS
        ----------
        - mask (torch.Tensor) : composite trunk mask (lungs + abdomen)
        - img_height (int) : height of original CXR
        - img_width (int) : width of original CXR
        - pad_fraction (float) : % padding to add to bounding box

        RETURNS
        -------
        - x_min, x_max, y_min, y_max (torch.Tensor) : bounding box coordinates
        """
        # mask coordinates
        y_indices, x_indices = torch.where(mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        if pad_fraction:
            # bounding box shape
            box_width = x_max - x_min + 1
            box_height = y_max - y_min + 1

            # padding amounts
            pad_x = int(pad_fraction * box_width)
            pad_y = int(pad_fraction * box_height)

            # adjust bounding box with padding, staying within image boundaries
            x_min = max(0, x_min - pad_x)
            x_max = min(img_width - 1, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(img_height - 1, y_max + pad_y)

        return x_min, x_max, y_min, y_max

    def crop_trunk(self, original_img, processed_img, pad_fraction, return_lungs=False):
        """
        Crops frontal CXRs around the trunk area.

        PARAMETERS
        ----------
        - original_img (numpy.ndarray) : original CXR
        - processed_img (torch.Tensor) : CXR processed for segmentation
        - pad_fraction (float) : % padding to add to bounding box

        RETURNS
        -------
        - cropped_img (numpy.ndarray) : original CXR cropped around the trunk
        - cropped_mask (torch.Tensor) : composite trunk mask downscaled to the cropped CXR's dimensions
        """
        with torch.no_grad(): # inference
            pred_prob = torch.sigmoid(self._model(processed_img)) # sigmoid activation function

        # scale maps to original image size
        height, width = original_img.shape
        scaled_pred_prob = F.interpolate(pred_prob, size=(height, width), mode='bilinear', align_corners=False)

        # binary lung masks
        mask_l, mask_r = scaled_pred_prob[0, 4] >= .5, scaled_pred_prob[0, 5] >= .5

        # check if lung masks exist
        if not mask_l.any() or not mask_r.any():
            raise ValueError('Lung mask(s) missing')

        # combine masks
        mask_abd = scaled_pred_prob[0, 10] >= .5 # facies diaphragmatica
        combined_mask = mask_l | mask_r | mask_abd

        # connected components: expecting, at most, 2 separate lungs + one abdomen
        _, n_components = label(combined_mask.cpu())
        if n_components > 3:
            raise ValueError(f'Abnormal number of masks detected - {n_components}')
            
        # composite mask bounding box
        x_min, x_max, y_min, y_max = self.mask_bounding_box(combined_mask, height, width, pad_fraction)

        # crop image and mask
        cropped_img = original_img[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = combined_mask[y_min:y_max+1, x_min:x_max+1]

        if return_lungs:
            cropped_mask_l = mask_l[y_min:y_max+1, x_min:x_max+1]
            cropped_mask_r = mask_r[y_min:y_max+1, x_min:x_max+1]

            return cropped_img, cropped_mask, cropped_mask_l, cropped_mask_r

        return cropped_img, cropped_mask

    def fill_mask(self, mask):
        """
        Fills the pixels between masks within the bounding box of the composite mask.

        PARAMETERS
        ----------
        - mask (torch.Tensor) : composite trunk mask downscaled to the cropped CXR's dimensions

        RETURNS
        -------
        - mask (torch.Tensor) : filled composite mask, modified in place
        """
        # composite mask bounding box
        x_min, x_max, y_min, y_max = self.mask_bounding_box(mask, None, None, None)

        # submask within bounding box
        submask = mask[y_min:y_max + 1, x_min:x_max + 1]

        H_sub, W_sub = submask.shape

        # grids for rows and columns
        y_grid = torch.arange(H_sub, device=self._device).unsqueeze(1)
        x_grid = torch.arange(W_sub, device=self._device).unsqueeze(0)

        # horizontal fill
        rows_with_true = submask.any(dim=1)
        # for each row, find leftmost and rightmost True pixels
        if rows_with_true.any():
            x_left = torch.where(rows_with_true, submask.float().argmax(dim=1), W_sub)
            x_right = torch.where(rows_with_true, W_sub - submask.flip(dims=(1,)).float().argmax(dim=1) - 1, -1)

            # expand to match grid dimensions
            x_left_expanded = x_left.unsqueeze(1)
            x_right_expanded = x_right.unsqueeze(1)

            # create mask for horizontal fill
            horizontal_fill = (x_grid >= x_left_expanded) & (x_grid <= x_right_expanded)
            submask |= horizontal_fill

        # vertical fill
        cols_with_true = submask.any(dim=0)
        if cols_with_true.any():
            y_top = torch.where(cols_with_true, submask.float().argmax(dim=0), H_sub)
            y_bottom = torch.where(cols_with_true, H_sub - submask.flip(dims=(0,)).float().argmax(dim=0) - 1, -1)

            y_top_expanded = y_top.unsqueeze(0)
            y_bottom_expanded = y_bottom.unsqueeze(0)

            vertical_fill = (y_grid >= y_top_expanded) & (y_grid <= y_bottom_expanded)
            submask |= vertical_fill

        return mask
    
    @staticmethod
    def border_connected_components(outlier_mask, min_size=100):
        """
        Retains connected components in a mask that touch the image borders.

        PARAMETERS
        ----------
        - outlier_mask (numpy.ndarray) : binary mask of outlier pixels (True = outlier)

        RETURNS
        -------
        - border_mask (numpy.ndarray) : binary mask with only border-connected components
        """
        # connected components in the outlier mask
        labeled_mask,_ = label(outlier_mask)
        component_sizes = np.bincount(labeled_mask.ravel())

        # components touching the borders
        border_labels = np.unique(np.concatenate([
            labeled_mask[0, :],    # top
            labeled_mask[-1, :],   # bottom
            labeled_mask[:, 0],    # left
            labeled_mask[:, -1],   # right
        ]))

        # exclude background label
        border_labels = border_labels[border_labels != 0]

        valid_labels = border_labels[component_sizes[border_labels] >= min_size]

        # mask with only border-connected components
        border_mask = np.isin(labeled_mask, valid_labels)
    
        return border_mask
    
    def find_outliers(self, img, roi_mask, outlier_thresh):
        """
        Identifies image background based on pixel intensity distribution.

        PARAMETERS
        ----------
        - img (numpy.ndarray) : original CXR cropped around the trunk
        - roi_mask (torch.Tensor) : composite trunk mask downscaled to the cropped CXR's dimensions
        - outlier_thresh (float) : threshold for the proportion of outlier pixels to decide whether to clip them

        RETURNS
        -------
        - outlier_mask (numpy.ndarray or None) : mask of outlier pixels if clipping is to be applied; otherwise, None
        """
        filled_mask = self.fill_mask(roi_mask) # fill gaps between masks
        
        # probability mass function
        hist, bin_centers = histogram(img, nbins=256, normalize=True)
        smoothed_hist = gaussian_filter1d(hist, sigma=1) # gaussian smoothing

        peak_indices,_ = find_peaks(smoothed_hist, prominence=1e-6)
        thresh = bin_centers[peak_indices[0]]

        outlier_mask = (img <= thresh)
        outlier_mask[filled_mask.cpu()] = False # no outliers within ROI

        outlier_mask = self.border_connected_components(outlier_mask)

        # keep outlier mask if the proportion of outlier pixels >= threshold
        outlier_prop = float(outlier_mask.sum() / (~filled_mask).sum())
        if outlier_prop >= outlier_thresh:
            return outlier_mask
        return None

    def export_failed_cxrs(self, output_path='failed_cxrs.csv'):
        """
        Exports failed_cxrs dictionary to .CSV file.

        PARAMETERS
        ----------
        - output_path (str) : The file path where the CSV will be saved (default = 'failed_cxrs.csv')
        """
        if not self.failed_cxrs:
            print('Nothing to export')
            return
    
        df = pd.DataFrame(list(self.failed_cxrs.items()), columns=['Image Path', 'Failure Reason'])
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f'Exported failed CXRs to {output_path}')

    def save_image(self, img, img_path):
        """
        Saves processed CXR as .PNG file.
        """
        name = os.path.basename(img_path)
        pre = os.path.dirname(img_path)

        name = name.split('.')[0] + '.png' if not name.endswith('.png') else name
        name = 'PROCESSED_' + name
        new_path = os.path.join(pre, name)
        
        cv2.imwrite(new_path, img)
        print(f'({self.__count:,}) Saved CXR as {new_path}')
        self.__count += 1

    def preprocess_cxr(self, img_path, pad_fraction=.025, outlier_thresh=.05, clipLimit=2, save=False):
        try:
            original, processed = self.load_and_process(img_path) # load image

            # crop trunk
            cropped_img, cropped_mask = self.crop_trunk(original, processed, pad_fraction=pad_fraction)

            # background pixels
            outlier_mask = self.find_outliers(cropped_img, cropped_mask, outlier_thresh=outlier_thresh)
            
            clahe_img = cv2.createCLAHE(clipLimit=clipLimit).apply(cropped_img) # enhance contrast
                
            if outlier_mask is not None:
                clahe_img[outlier_mask] = 0 # clip background pixels to 0

            if not save:
                return clahe_img

            self.save_image(clahe_img, img_path)

        except ValueError as e:
            print(f'\nError processing CXR: {e}. Image path saved to failed_cxrs.\n')
            self.failed_cxrs[img_path] = str(e)


def get_chex_img_path(cheXbert_path:str) -> str:
    path = '/home/freddie/CheX_Data/chexpertchestxrays-u20210408/CheXpert-v1.0'
    
    patient_num = int(cheXbert_path[27:32])
    if 1 <= patient_num <= 21513:
        path += ' batch 2 (train 1)/'
    elif 21514 <= patient_num <= 43017:
        path += ' batch 3 (train 2)/'
    elif 43018 <= patient_num <= 64540:
        path += ' batch 4 (train 3)/'
    else:
        path += ' batch 1 (validate & csv)/valid/'

    return path + cheXbert_path[20:]

get_pad_img_path = lambda pad_row : f'/home/freddie/BIMCV-PadChest-FULL/{pad_row["ImageDir"]}/{pad_row["ImageID"]}'

get_vindr_img_path = lambda vindr_row:f'/home/freddie/VinDR_Data/physionet.org/files/vindr-cxr/1.0.0/{vindr_row["Set"]}/{vindr_row["image_id"]}.dicom'

get_rsna_img_path = lambda patientId : f'/home/freddie/RSNA_Data/stage_2_train_images/{patientId}.dcm'

def processed_file_name(img_path):
    name = os.path.basename(img_path)
    pre = os.path.dirname(img_path)

    name = name.split('.')[0] + '.png' if not name.endswith('.png') else name
    name = 'PROCESSED_' + name
    return os.path.join(pre, name)

if __name__ == '__main__':
    #chex = pd.read_csv('/home/freddie/CheX_Data/chexpertchestxrays-u20210408/train_cheXbert_w_views.csv')
    #pad = pd.read_csv('/home/freddie/BIMCV-PadChest-FULL/PADCHEST_chest_x_ray_images_pneumonia.csv')
    #mimic = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-pneumonia-frontal.csv')
    #mimic_llm = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-pneumonia-full-wreports_llm_complete_frontal.csv')
    #vindr = pd.read_csv('/home/freddie/VinDR_Data/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_merged_pneumonia.csv')
    rsna = pd.read_csv('/home/freddie/RSNA_Data/stage_2_train_labels.csv')

    # remove abnormal cxrs, only frontal images, and no empty pneumonia values
    #chex = chex[(chex.ViewAgreement == True) & (chex['Frontal/Lateral'] == 'Frontal') & (~chex.Pneumonia.isna())]
    #paths = chex.Path.apply(get_chex_img_path).tolist()
    
    #paths = pad.apply(get_pad_img_path, axis=1).tolist()
    
    #paths = mimic.Path.tolist()
    #paths = [img for img in mimic_llm.Path if not os.path.isfile(processed_file_name(img))]
    
    #paths = vindr.apply(get_vindr_img_path, axis=1).tolist()

    paths = rsna.patientId.apply(get_rsna_img_path).tolist()

    processor = CXR_Processor()
    for path in paths:
        processor.preprocess_cxr(path, save=True)

    processor.export_failed_cxrs('/home/freddie/RSNA_Data/RSNA_failed_cxrs.csv')
