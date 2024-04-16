import os
import shutil
from glob import glob

train_pred_dir = 'outputs/sam_feat_selector__motion_pred_v3--MOT17-images/test'
test_pred_dir = 'outputs/sam_feat_selector__motion_pred_v3--MOT17-images/train'
final_pred_dir = train_pred_dir.replace('test', 'final')
os.makedirs(final_pred_dir, exist_ok=True)

#     # '''for MOT17 submit'''
#  Copy train pred and test pred to final pred and duplicate 2 more files with replaced suffix (-SDP to -DPM and to -FRCNN)

for file in glob(f'{train_pred_dir}/*.txt'):
    shutil.copy(file, file.replace(train_pred_dir, final_pred_dir))
    shutil.copy(file, file.replace(train_pred_dir, final_pred_dir).replace('-SDP', '-DPM'))
    shutil.copy(file, file.replace(train_pred_dir, final_pred_dir).replace('-SDP', '-FRCNN'))

for file in glob(f'{test_pred_dir}/*.txt'):
    shutil.copy(file, file.replace(test_pred_dir, final_pred_dir))
    shutil.copy(file, file.replace(test_pred_dir, final_pred_dir).replace('-SDP', '-DPM'))
    shutil.copy(file, file.replace(test_pred_dir, final_pred_dir).replace('-SDP', '-FRCNN'))