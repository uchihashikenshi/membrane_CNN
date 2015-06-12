# coding:utf-8
from PIL import Image
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from theano.tensor.signal import downsample
import theano.tensor as T
import theano

memCNN_home = os.getcwd() # projects/memCNNから走らせる予定

# ✕train ◯training

"""
Annotation
data_type: trainingとかtestとか
data_dir: データを入れてるディレクトリの名前。data/raw/(data_type)/まではprefix
"""
class Preprocessing(object):
    def __init__(self):
        pass

    def load_images(self, data_type, data_dir):
        os.chdir("data/raw/%s" % data_dir)
        filelist = glob.glob('*.tif') # とりあえずtifのみ対応
        return filelist

    def image_to_array(self, file):
        raw_image = Image.open(file)
        raw_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
        return raw_matrix

    def median_extract(self, data_type, data_dir):
        filelist = self.load_images(data_type, data_dir)
        if os.path.exists("%s/data/%s_dataset/median_extract_%s_dataset" % (memCNN_home, data_type, data_type)) != True:
            os.mkdir("%s/data/%s_dataset/median_extract_%s_dataset" % (memCNN_home, data_type, data_type)) # データ置き場用意
        # スタック中全画像からmedianを求める(medianの平均値)
        # fixme: medianの平均でいいのか・・・？
        N, _sum = 0, 0
        for file in filelist:
            raw_matrix = self.image_to_array(file)
            median = np.median(raw_matrix)
            _sum += median
            N += 1
        stack_median = _sum / N

        file_num = 1
        for file in filelist:
            raw_matrix = self.image_to_array(file)
            median = np.median(raw_matrix) #中央値
            # スタックのmedianに各画像のmedianを合わせる
            median_extract_matrix = (raw_matrix - (median - stack_median))
            # 負の画素値を0に補正
            # fixme: こんな処理を入れずにスマートにやりたい
            for i in range(1024):
                for j in range(1024):
                    if median_extract_matrix[i][j] < 0:
                        median_extract_matrix[i][j] = 0
            median_extract_image = Image.fromarray(np.uint8(median_extract_matrix).reshape(1024, 1024))
            median_extract_image.save("%s/data/%s_dataset/median_extract_%s_dataset/median_extract_image_%03d.tif" % (memCNN_home, data_type, data_type, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print("%s epoch ended" % file_num)
        print("median_extract_%s_dataset is created" % data_type)

    def make_average_pooled_image(self, data_dir):
        filelist = self.load_images(data_dir)

        if os.path.exists("%s/data/raw/pooled_dataset" % memCNN_home) != True:
            os.mkdir("%s/data/raw/pooled_dataset" % memCNN_home) # データ置き場用意

        file_num = 1
        for file in filelist:
            raw_image = Image.open(file)
            raw_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
            pool_size = (4, 4) # とりあえずベタ書き
            pooled_matrix = []
            for i in range(int(1024 / 4)):
                for j in range(int(1024 / 4)):
                    _sum = 0
                    for k in range(4):
                        for l in range(4):
                            _sum += raw_matrix[4 * i + k, 4 * j + l]
                    pooled_pixel = round(_sum / 16)
                    pooled_matrix.append(pooled_pixel)
            pooled_image = Image.fromarray(np.uint8(pooled_matrix).reshape(256, 256))
            pooled_image.save("%s/data/raw/pooled_dataset/pooled_image_%03d.tif" % (memCNN_home, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print("%s epoch ended" % file_num)
        print("train 100 raw tif dataset is created")
