import numpy as np
from tqdm import tqdm
import cv2
import os
from os.path import join

def crop_image(img, bbox = [0,2848,274,3730], square=True):
    y1,y2,x1,x2 = bbox
    # crop image
    crop_img = img[y1:y2, x1:x2]
    if square:
        # find the minimum square size for the cropped image
        min_size = max(crop_img.shape[0], crop_img.shape[1])
        # create an empty image with the minimum square size
        if len(crop_img.shape) == 2:
            empty_img = np.zeros((min_size, min_size), dtype=np.uint8)
        else:
            empty_img = np.zeros((min_size, min_size, 3), dtype=np.uint8)
        # fill the empty image with the cropped image
        # cropped image should be in the center of the empty image
        diff_of_dimensions = abs(crop_img.shape[0] - crop_img.shape[1])
        if crop_img.shape[0] > crop_img.shape[1]:
            empty_img[:, diff_of_dimensions//2:diff_of_dimensions//2+crop_img.shape[1]] = crop_img
        else:
            empty_img[diff_of_dimensions//2:diff_of_dimensions//2+crop_img.shape[0], :] = crop_img
        return empty_img
    else:
        return crop_img

def zoom(img_path, label_path, img_savedir, label_savedir, mixlabel_savedir, vessel_savedir, od_savedir):
    if not os.path.exists(img_savedir):
        os.makedirs(img_savedir)
    if not os.path.exists(label_savedir):
        os.makedirs(label_savedir)
        os.makedirs(label_savedir + '/EX')
        os.makedirs(label_savedir + '/HE')
        os.makedirs(label_savedir + '/MA')
        os.makedirs(label_savedir + '/SE')
        os.makedirs(label_savedir + '/BV')
        os.makedirs(label_savedir + '/OD')
    if not os.path.exists(mixlabel_savedir):
        os.makedirs(mixlabel_savedir)
    if not os.path.exists(vessel_savedir):
        os.makedirs(vessel_savedir)
    if not os.path.exists(od_savedir):
        os.makedirs(od_savedir)
    fileList = os.listdir(img_path)
    for jpgfile in tqdm(fileList):
        (realname, extension) = os.path.splitext(jpgfile)
        if extension != '.jpg' and extension != '.png':
            continue
        img = cv2.imread(os.path.join(img_path, jpgfile))
        label1 = cv2.imread(os.path.join(label_path, 'EX', realname + '_EX.tif'), 0)
        label2 = cv2.imread(os.path.join(label_path, 'HE', realname + '_HE.tif'), 0)
        label3 = cv2.imread(os.path.join(label_path, 'MA', realname + '_MA.tif'), 0)
        label4 = cv2.imread(os.path.join(label_path, 'SE', realname + '_SE.tif'), 0)
        label_all = np.zeros_like(img)
        label_all[np.where(label1 > 0)] = 1
        label_all[np.where(label2 > 0)] = 2
        label_all[np.where(label3 > 0)] = 3 # butun masklari okuyup bos bi arraye 1 2 3 4 olarak yaziyor
        label_all[np.where(label4 > 0)] = 4
        try:
            label5 = cv2.imread(os.path.join(label_path, 'BV', realname + '_BV.tif'), 0)
            # check if the image is read
            if label5.size ==0:
                raise Exception('Image not found.')
        except:
            label5 = cv2.imread(os.path.join(label_path, 'BV', realname + '_BV.png'), 0)
        label6 = cv2.imread(os.path.join(label_path, 'OD', realname + '_OD.tif'), 0)
        crop_img = crop_image(img)
        crop_label1 = crop_image(label1)
        crop_label2 = crop_image(label2)
        crop_label3 = crop_image(label3)
        crop_label4 = crop_image(label4)
        crop_label_all = crop_image(label_all)
        crop_label5 = crop_image(label5)
        crop_label6 = crop_image(label6)

        cv2.imwrite(os.path.join(img_savedir, realname + '.png'), crop_img)
        cv2.imwrite(os.path.join(label_savedir, 'EX', realname + '.png'), crop_label1)
        cv2.imwrite(os.path.join(label_savedir, 'HE', realname + '.png'), crop_label2)
        cv2.imwrite(os.path.join(label_savedir, 'MA', realname + '.png'), crop_label3)
        cv2.imwrite(os.path.join(label_savedir, 'SE', realname + '.png'), crop_label4)
        cv2.imwrite(os.path.join(mixlabel_savedir, realname + '.png'), crop_label_all)
        cv2.imwrite(os.path.join(label_savedir, 'BV', realname + '.png'), crop_label5)
        cv2.imwrite(os.path.join(label_savedir, 'OD', realname + '.png'), crop_label6)
        cv2.imwrite(os.path.join(vessel_savedir, realname + '.png'), crop_label5)
        cv2.imwrite(os.path.join(od_savedir, realname + '.png'), crop_label6)

def preprocess_dataset(original_dataset, preprocessed_dataset):
    
    for f in ['train', 'test']:
        img_path = os.path.join(original_dataset,f'image/{f}')
        label_path = os.path.join(original_dataset,f'label/{f}')
        if f == 'test':
            f = 'val'
        img_savedir = os.path.join(preprocessed_dataset, f, '4 classes/image_zoom_hd')
        label_savedir = os.path.join(preprocessed_dataset, f, 'label_zoom_hd')
        mixlabel_savedir = os.path.join(preprocessed_dataset, f, '4 classes/label_zoom_hd')
        vessel_savedir = os.path.join(preprocessed_dataset, f, '4 classes/vessel_mask_zoom_hd')
        od_savedir = os.path.join(preprocessed_dataset, f, '4 classes/od_mask_zoom_hd')
        zoom(img_path, label_path, img_savedir, label_savedir, mixlabel_savedir, vessel_savedir, od_savedir)

adem_path = '/home/adem/Desktop/Thesis/Codes/Retinal-lesion-Segmentation-main/IDRiD'