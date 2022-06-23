# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

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
import os

    # file exists
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir='s:\good_imgs/'
    file_client = FileClient('disk')
    opt_path = dir_path+'/options/test/REDS/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)
    filenames = next(walk(dir), (None, None, []))[2]
    for fname in filenames:
        if fname.endswith("_raw.jpg"):
            my_file = Path(dir+fname[:-8]+".jpg")
            if not my_file.is_file():
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


if __name__ == '__main__':
    main()

