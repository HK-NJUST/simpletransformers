'''
Author: Kai.Han
Date: 2021-10-25 15:26:13
LastEditTime: 2022-07-11 18:13:43
'''
# -*- coding:utf-8 -*- 
import os
from re import L
import cv2
import json
import httpx
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool


HTTP_CLI = httpx.Client(timeout=None)

l1_l2 = {}
l1s = []
l2s = []
l2_idx = {}

def parse_l1l2(path, root):
    df = pd.read_csv(path)
    res = []
    for i, row in df.iterrows():
        l1 = row['L1'].strip()
        l2 = row['L2'].strip()
        l1s.append(l1)
        l2s.append(l2)
        l2_idx[l2] = i
        res.append(f"{l1}\t{l2}\n")
        if l1 in l1_l2:
            l1_l2[l1].append(l2)
        else:
            l1_l2[l1] = [l2]
    root_up = os.path.dirname(root)
    with open(os.path.join(root_up, "l1_l2_records.txt"), 'w') as fo:
        fo.writelines(res)


def read_one_json(path):
    f = open(path, 'r')
    return json.load(f)


def get_im_from_hash(im_hash: str) -> bytes:
    if im_hash.startswith("https"):
        r = HTTP_CLI.get(im_hash)
    else:
        r = HTTP_CLI.get(f"http://cf.shopee.co.id/file/{im_hash}")
        
    if r.status_code != 200:
        print("bad image hash: ", im_hash)
        return []
    img = np.frombuffer(r.content, dtype=np.int8)
    img = cv2.imdecode(img, 1) # bgr
    if isinstance(img, type(None)):
        return None

    return img
    

def parse_gt(lines):
    count = 0
    results = []
    for idx, line in enumerate(lines):
        count += 1
        url = line['data']["image"]
        hash = url.split('/')[-1]
        results.append(hash)
    return results


def read_txt(file):
    with open(file, 'r') as fo:
        return [line.strip().split('\t')[-1] for line in fo.readlines()]


def download_func(lines, img_dir, failure_file):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        os.makedirs(os.path.join(img_dir, "labels"))
        os.makedirs(os.path.join(img_dir, "images"))

    imgnames = os.listdir(os.path.join(img_dir, "images"))
    for idx, row in tqdm(lines.iterrows()):
        try:
            ins = {}
            if isinstance(row['cover_url'], float):
                cover_url = ""
            else:
                cover_url = row['cover_url'].replace('"', '')
            if not cover_url.startswith('https'):
                print('bad_url', cover_url)
                continue
            if isinstance(row['text'], float):
                text = ''
            else:
                text = row['text'].replace('"', '')
            maunal_labels = row['maunal_labels']
            label_json = json.loads(maunal_labels)
            labels = label_json['admin_ext_detail_list']
            l1, l2 = None, None
            for label in labels:
                if label['label_info'][0]['label_type_name'] != 'Content Label':
                    continue
                l1 = label['label_info'][0]['name']
                if 'children' not in label['label_info'][0]:
                    continue
                l2 = label['label_info'][0]['children'][0]['name']
                if l2 not in l2_idx:
                    print("l2:", l2)
                    continue
                new_idx = l2_idx[l2]
                l1_l2 = f"{l1}#{l2}"
                ins['labels'] = str(new_idx)
                ins['l1#l2'] = l1_l2
            ins['text'] = text

            hash = os.path.basename(cover_url)
            if hash.endswith('.jpg'):
                imgname = hash
            else:
                imgname = os.path.basename(hash) + '.jpg'
            if imgname in imgnames:
                continue
            if (idx + 1) % 50 == 0:
                print(idx + 1)
            
            img = get_im_from_hash(cover_url)
            if img.any() == None:
                print("none img")
                continue
            if 'labels' not in ins:
                print(row)
                continue
            cv2.imwrite(f"{os.path.join(img_dir, 'images', imgname)}", img)
            file_name = os.path.join(img_dir, "labels", imgname.replace('.jpg', '.json'))
            with open(file_name, 'w') as file:
                json.dump(ins, file, indent=4)
        except Exception as e:
            print(e)
            with open(failure_file, 'w') as fo:
                fo.writelines(hash + '\n')


def multi_processing(df, process_num, img_dir, failure_file):
    df_splits = np.array_split(df, process_num)
    p = Pool(process_num)

    for df_ins in df_splits:
        p.apply_async(download_func, args=(df_ins, img_dir, failure_file)) 
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == "__main__":
    label_path = "/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_train_01_240820_240901.csv"
    l1l2_path = "/home/work/video_hk/projects/24q3/simpletransformers/data/sv_tag_cls.csv"
    failure_file = "/home/work/video_hk/projects/24q3/simpletransformers/data/failure_hash.txt"
    img_dir = "/home/work/video_hk/data/video_tag/train_240820_240901"
    
    parse_l1l2(l1l2_path, img_dir)
    process_num = 10

    # lines = read_one_json(label_path)
    # lines = parse_gt(lines)
    # lines = read_txt(label_path)
    df = pd.read_csv(label_path)
    from random import sample
    # lines = sample(lines, 500)
    multi_processing(df, process_num, img_dir, failure_file)

    # root = os.path.dirname(img_dir)
    # input_file = os.path.join(root, 'all_img_list.json')
