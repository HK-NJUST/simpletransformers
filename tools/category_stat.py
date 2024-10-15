import os
import json
import pandas as pd
from collections import defaultdict


test_file = "/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_test_240731.csv"
test_dir = "/home/work/video_hk/data/video_tag/test_240731/labels"
l1l2 = "/home/work/video_hk/projects/24q3/simpletransformers/outputs/l1_l2_records.txt"
pr_stat = "/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_stat.txt"
out_file = "/home/work/video_hk/projects/24q3/simpletransformers/data/pr_stat.txt"

test_labels = os.listdir(test_dir)

# idx_l12 = {}
# with open(l1l2, 'r') as fin:
#     lines = fin.readlines()
#     for idx, line in enumerate(lines):
#         # l1, l2 = line.strip().split('\t')
#         idx_l12[idx] = line.strip()

idx_c = defaultdict(int)
for name in test_labels:
    test_json = os.path.join(test_dir, name)
    res = json.load(open(test_json, 'r'))
    idx = res['labels']
    l12 = res['l1#l2']
    idx_c[idx] += 1
    res

sorted_dict_desc = dict(sorted(idx_c.items(), key=lambda item: item[1], reverse=True))
print(sorted_dict_desc)

all_res = []
with open(pr_stat, 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        b_ = line.strip().split('\t')
        idx = b_[0]
        if idx not in sorted_dict_desc:
            continue
        num = sorted_dict_desc[idx]
        new_line = line.replace('\n', f"\t{num}\n")
        all_res.append(new_line)

with open(out_file, 'w') as fo:
    fo.writelines(all_res)