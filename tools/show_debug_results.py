import os
import pandas as pd


if __name__ == "__main__":
    test_file = "/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_test_240731.csv"
    results = "/home/work/video_hk/projects/24q3/simpletransformers/outputs/test_240731_eval_0925_e40/debug_show.txt"
    l1l2 = "/home/work/video_hk/projects/24q3/simpletransformers/outputs/l1_l2_records.txt"
    out_file = "/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_test_240731.txt"

    idx_l12 = {}
    with open(l1l2, 'r') as fin:
        lines = fin.readlines()
        for idx, line in enumerate(lines):
            # l1, l2 = line.strip().split('\t')
            idx_l12[idx] = line.strip()

    df = pd.read_csv(test_file)
    hash_info_dict = {}
    for idx, row in df.iterrows():
        cid, vurl, text, curl = row[:4]
        chash = os.path.basename(curl) + '.jpg'
        hash_info_dict[chash] = [vurl, text, curl]

    new_outs = []
    with open(results, 'r') as fin:
        lines = fin.readlines()
        for idx, line in enumerate(lines):
            b_ = line.strip().split()
            pidx, gt_id = int(b_[-2]), int(b_[-1])
            pred_name, gt_name = idx_l12[pidx], idx_l12[gt_id]
            imgname = os.path.basename(line.strip().split('\t')[0])
            if imgname in hash_info_dict:
                new_line = "\t".join(map(str, hash_info_dict[imgname])) + '\t' + line.strip() + f"\t{pred_name}\t{gt_name}\n"
                new_outs.append(new_line)
    
    with open(out_file, 'w') as fo:
        fo.writelines(new_outs)
