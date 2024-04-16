import os

TRACKER_PATH = 'outputs/sam_feat_selector__motion_pred_v3--MOT17-images/final'

def shift_obj_id(path, shift):
    with open(path, 'r') as f:
        lines = f.readlines()
    with open(path, 'w') as f:
        for line in lines:
            line = line.strip().split(',')
            line[1] = str(int(line[1]) + shift)
            f.write(','.join(line) + '\n')

for file in os.listdir(TRACKER_PATH):
    if file.endswith('.txt'):
        shift_obj_id(os.path.join(TRACKER_PATH, file), 1)