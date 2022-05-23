# import libraries and package
import os
import math
import itertools
import multiprocessing
import pandas
import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from time import time
from collections import OrderedDict
gt0 = time()

import warnings
warnings.filterwarnings('ignore')

###########################################################
# import KDD dataset 
train20_nsl_kdd_dataset_path = os.path.join("KDD_files", "KDDTrain+_20Percent.txt")
train_nsl_kdd_dataset_path = os.path.join("KDD_files", "KDDTrain+.txt")
test_nsl_kdd_dataset_path = os.path.join("KDD_files", "KDDTest+.txt")

###########################################################
# header_names is a list of feature names in the same order as the data
header_names = nu.array(['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred','attack_category_label'])

###########################################################
# map attack type to attack category
category = defaultdict(list)
category['benign'].append('normal')

with open('KDD_files/training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v,k) for k in category for v in category[k])
print(attack_mapping)

###########################################################
# Separate between nominal  binary and numerical features.
nominal_inx = [1, 2, 3]
binary_inx = [6, 11, 13, 14, 20, 21]
numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))

nominal_cols = header_names[nominal_inx].tolist()
binary_cols = header_names[binary_inx].tolist()
numeric_cols = header_names[numeric_inx].tolist()



