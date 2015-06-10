# coding:utf-8
from PIL import Image
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from theano.tensor.signal import downsample
import theano.tensor as T
import theano

memCNN_home = os.getcwd() # projects/memCNNから走らせる予定

class Preprocessing():
    def __init__(self):
        pass

    def load_images(self, data_dir):
        os.chdir("data/raw/%s" % data_dir)
        filelist = glob.glob('*.tif') # とりあえずtifのみ対応
        return filelist

    def median_extract(self):
        pass

    def make_average_pooled_image(self, data_dir):
        filelist = self.load_images(data_dir)

        if os.path.exists("%s/data/raw/pooled_dataset" % memCNN_home) != True:
            os.mkdir("%s/data/raw/pooled_dataset" % memCNN_home) # データ置き場用意

        file_num = 1
        for file in filelist:
            raw_image = Image.open(file)
            raw_pixel_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
            pool_size = (4, 4) # とりあえずベタ書き
            pooled_pixel_matrix = []
            for i in range(int(1024 / 4)):
                for j in range(int(1024 / 4)):
                    _sum = 0
                    for k in range(4):
                        for l in range(4):
                            _sum += raw_pixel_matrix[4 * i + k, 4 * j + l]
                    pooled_pixel = round(_sum / 16)
                    pooled_pixel_matrix.append(pooled_pixel)
            pooled_image = Image.fromarray(np.uint8(pooled_pixel_matrix).reshape(256, 256))
            plt.imshow(pooled_image)
            pooled_image.save("%s/data/raw/pooled_dataset/pooled_image_%03d.tif" % (memCNN_home, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print("%s epoch ended" % file_num)
        print("train 100 raw tif data created")
