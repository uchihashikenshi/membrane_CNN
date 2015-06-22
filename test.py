# coding:utf-8
from utils import Preprocessing

p = Preprocessing()
#p.make_median_extracted_dataset("training", "raw/train-input")
#p.make_median_extracted_dataset("test", "raw/test-input")
#p.make_average_pooled_dataset("training", "preprocessed/training/median_extract_training_dataset")
#p.make_average_pooled_dataset("test", "preprocessed/test/median_extract_test_dataset")
p.patch_extract("preprocessed/training/pooled_training_dataset", "raw/train-labels", "separated_")
