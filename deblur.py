# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json

import csv
import numpy as np
from datetime import datetime
import PIL
from timm import timm_create_model

import torch as th
import torchvision.transforms as T

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model

from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
from os import walk
# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str
import glob
from basicsr.utils.options import parse
from pathlib import Path
def create_convnext_model():
    f = open("s:/content/labels.csv", 'a', newline='')
    writer = csv.writer(f)
    string = str(datetime.now()).replace(" ", "|")
    string = np.append(string, " ---start-----")
    writer.writerow(string)
    f.close()
    ordinal='0'
    if th.cuda.device_count() > 1:
      print("Let's use", th.cuda.device_count(), "GPUs!")
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    device = th.device('cuda:{}'.format(ordinal))
    model_name = "convnext_xlarge_in22k"
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device for convnext = ", device)
    #device='cuda'
    # create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
    model = timm_create_model(model_name, pretrained=True)
    model.to(device)

    # Define transforms for test
    from timm.data.constants import \
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
    NORMALIZE_STD = IMAGENET_DEFAULT_STD
    SIZE = 256

    # Here we resize smaller edge to 256, no center cropping
    transforms = [
                  T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
                  T.ToTensor(),
                  T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                  ]

    transforms = T.Compose(transforms)
    return model,transforms,device
def get_description(fname,convnext_model,transforms):
    img = PIL.Image.open(fname)
    img_tensor = transforms(img).unsqueeze(0).to(convnext_device)

    # inference
    output = th.softmax(convnext_model(img_tensor), dim=1)
    top5 = th.topk(output, k=5)

    return top5
def check_image_by_convnext(filename,convnext_model,transforms):
    matches = ["pen", "ink","sketch"]
    top5 = get_description(filename, convnext_model, transforms)
    # blablabla
    top5_prob = top5.values[0]
    top5_indices = top5.indices[0]
    string = filename
    for i in range(5):
        labels = imagenet_labels[str(int(top5_indices[i]))]
        prob = "{:.2f}%".format(float(top5_prob[i]) * 100)
        print(labels, prob)
        string = np.append(string, labels)
        string = np.append(string, prob)
    f = open("s:/content/labels.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(string)
    f.close()
    for i in range(5):
        labels = imagenet_labels[str(int(top5_indices[i]))]
        pr=float(top5_prob[i]) * 100

        prob = "{:.2f}%".format(float(top5_prob[i]) * 100)
        if pr<10 :
            return 0
        else:
            if any(x in labels for x in matches):
                return 0
            else:
                return pr
        print(labels, prob)
        string = np.append(string, labels)
        string = np.append(string, prob)
    # file exists

dir_path = os.path.dirname(os.path.realpath(__file__))
dir='s:\good_imgs/'
file_client = FileClient('disk')
imagenet_labels = json.load(open(dir_path+'/label_to_words.json'))
convnext_model, transforms, convnext_device = create_convnext_model()

filenames = next(walk(dir), (None, None, []))[2]
for fname in filenames:
    if fname.endswith("_raw.jpg"):
        my_file = Path(dir+fname[:-8]+".jpeg")
        if not my_file.is_file():
            opt_path = dir_path + '/options/test/REDS/NAFNet-width64.yml'
            opt = parse(opt_path, is_train=False)
            opt['dist'] = False
            model = create_model(opt)
            img_bytes = file_client.get(dir+fname, None)
            img = imfrombytes(img_bytes, float32=True)

            img = img2tensor(img, bgr2rgb=True, float32=True)

            ## 2. run inference

            model.feed_data(data={'lq': img.unsqueeze(dim=0)})

            if model.opt['val'].get('grids', False):
                model.grids()

            model.test()

            if model.opt['val'].get('grids', False):
                model.grids_inverse()

            visuals = model.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            imwrite(sr_img, str(my_file))
            print(str(my_file))
            rc = check_image_by_convnext(str(dir+fname),
                                         convnext_model, transforms)
            rc = check_image_by_convnext(str(my_file),
                                         convnext_model, transforms)



