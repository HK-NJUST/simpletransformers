


all_map = "/home/work/video_hk/data/video_tag/l1_l2_records.txt"
idx_l1l2 = {}
with open(all_map, 'r') as fo:
    lines = fo.readlines()

for idx, line in enumerate(lines):
    b_ = line.strip()
    idx_l1l2[idx] = b_
    b_


numbers = [86, 43, 13, 14, 71, 57, 44, 5, 16, 22, 29, 3, 37, 7, 39, 18, 66, 52, 42, 9]
outs = []
idx_newidx = {}
# for idx, num in enumerate(numbers):
#     txt = idx_l1l2[num]+'\n'
#     outs.append(txt)

oldidx_newidx = {}
for idx, num in enumerate(numbers):
    oldidx_newidx[num] = idx

for idx in idx_l1l2:
    if idx in numbers:
        continue
    oldidx_newidx[idx] = 0

for old, new in oldidx_newidx.items():
    outs.append(f"{old}\t{new}\n")
    
with open(all_map.replace("records", "records_idx_v2"), 'w') as fo:
    fo.writelines(outs)
