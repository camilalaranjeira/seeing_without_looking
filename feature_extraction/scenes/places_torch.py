import torch
from torch.autograd import Variable as V
import torchvision.models as models
import skimage.io
from torchvision import transforms as trn
from torch.nn import functional as F
import os, sys
import numpy as np
import cv2
# function to load exif of image
from PIL import Image, ExifTags

def imreadRotate(fn):
    image=Image.open(fn)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        print('dont rotate')
        pass
    return image

def load_labels(rootpath='.'):
    # prepare all the labels
    # scene category relevant
    file_name_category = os.path.join(rootpath, 'categories_places365.txt')
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url + f' -O {rootpath}')
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = os.path.join(rootpath, 'IO_places365.txt')
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url + f' -O {rootpath}')
    with open(file_name_IO) as f:
        lines = f.readlines()
        classes_macro = {}
        for line in lines:
            items = line.rstrip().split()
            classes_macro[items[0]] = (int(items[1])-1, int(items[2])-1)
            # labels_IO1.append(int(items[1]) -1) # 0 is indoor, 1 is outdoor
            # labels_IO2.append(int(items[2]) -1) # 0 is residential/nature, 1 is commercial/urban
    # labels_IO1 = np.array(labels_IO1)
    # labels_IO2 = np.array(labels_IO2)

    # # scene attribute relevant
    # file_name_attribute = 'labels_sunattribute.txt'
    # if not os.access(file_name_attribute, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
    #     os.system('wget ' + synset_url)
    # with open(file_name_attribute) as f:
    #     lines = f.readlines()
    #     labels_attribute = [item.rstrip() for item in lines]
    # file_name_W = 'W_sceneattribute_wideresnet18.npy'
    # if not os.access(file_name_W, os.W_OK):
    #     synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
    #     os.system('wget ' + synset_url)
    # W_attribute = np.load(file_name_W)

    return classes, classes_macro#, labels_attribute, W_attribute

# def hook_feature(module, input, output):
#     features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model(rootpath='.', device='cpu'):
    # this model has a last conv feature map as 14x14
    sys.path.append(rootpath)

    model_file = os.path.join(rootpath, 'whole_wideresnet18_places365.pth.tar')
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file  + f' -O {rootpath}')
        os.system(f'wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py -O {rootpath}')

    model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu
    model.eval()
    # hook the feature extractor
    # features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    # for name in features_names:
    #     model._modules.get(name).register_forward_hook(hook_feature)
    return model

