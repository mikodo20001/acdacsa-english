import random
import re

import os
from collections import defaultdict

random.seed(0)

labels = ['place', 'price', 'staff', 'miscellaneous', 'ambience', 'food', 'service', 'menu']
data_dir = "./MAMS"
output_dir = "./MAMS_acd_data"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_file_path = os.path.join(data_dir, "MAMS_train.txt")
valid_file_path = os.path.join(data_dir, "MAMS_val.txt")
test_file_path = os.path.join(data_dir, "MAMS_test.txt")

train_out_file_path = os.path.join(output_dir, "MAMS_train.txt")
valid_out_file_path = os.path.join(output_dir, "MAMS_val.txt")
test_out_file_path = os.path.join(output_dir, "MAMS_test.txt")

# 训练集
sent_labels = defaultdict(list)
with open(train_file_path, "r") as f:
    file = f.readlines()
    for line in file:
        x, y = line.split("\001")[0], line.strip().split("\001")[1]
        y = re.findall("The sentiment polarity of ([\s\S]*) is", y)[0]
        sent_labels[x].append(y)
with open(train_out_file_path, "w+") as f:
    for sent, cur_labels in sent_labels.items():
        for l in labels:
            pre_label = f"The {l} category is "
            if l in cur_labels:
                f.write(sent + "\001" + pre_label + "discussed .\n")
                continue
            f.write(sent + "\001" + pre_label + "not discussed .\n")

# 验证集
sent_labels = defaultdict(list)
with open(valid_file_path, "r") as f:
    file = f.readlines()
    for line in file:
        sent, cate, sentiment = line.split("\001")
        sent_labels[sent].append(cate)
with open(valid_out_file_path, "w+") as f:
    for sent, cur_labels in sent_labels.items():
        for l in labels:
            if l in cur_labels:
                f.write(sent + "\001" + l + "\001" + "discussed\n")
                continue
            f.write(sent + "\001" + l + "\001" + "not discussed\n")


# 测试集
sent_labels = defaultdict(list)
with open(test_file_path, "r") as f:
    file = f.readlines()
    for line in file:
        sent, cate, sentiment = line.split("\001")
        sent_labels[sent].append(cate)
with open(test_out_file_path, "w+") as f:
    for sent, cur_labels in sent_labels.items():
        for l in labels:
            if l in cur_labels:
                f.write(sent + "\001" + l + "\001" + "discussed\n")
                continue
            f.write(sent + "\001" + l + "\001" + "not discussed\n")
