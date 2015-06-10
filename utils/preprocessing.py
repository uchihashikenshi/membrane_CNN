# coding:utf-8
from PIL import Image
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from theano.tensor.signal import downsample
import theano.tensor as T
import theano

class Preprocessing(object):
    def __init__(self):
        pass

    def load_images(self):
        abs_path = os.getcwd() # projects/memCNNから走らせる予定
        print(abs_path)
        os.chdir("../")
        print(os.getcwd())

    def make_average_pooled_image(self):
        pass
