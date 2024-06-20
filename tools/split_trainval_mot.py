import os
import random
import shutil
from tqdm import tqdm

# Set the path to the MOT17 dataset directory
mot17_dir = 'data/Dataset/mot/MOT17/images/train'

# Set the path to the directory where you want to save the train and val sets
output_dir = 'data/Dataset/mot/MOT17_split/images/'

# Set the percentage of data to be used for validation
val_percentage = 0.2

# Create the train and val directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get the list of sequences in the MOT17 dataset
sequences = os.listdir(mot17_dir)

# Shuffle the sequences
random.shuffle(sequences)

# Split each video into two parts
for sequence in tqdm(sequences):
    sequence_dir = os.path.join(mot17_dir, sequence)
    images_dir = os.path.join(sequence_dir, 'img1')
    frames = os.listdir(images_dir)
    # Sort the frames
    frames.sort(key=lambda x: int(x.split('.')[0]))
    num_frames = len(frames)
    split_index = int(num_frames * 0.8)  # 80% for training, 20% for validation

    # Move the frames to the train directory
    train_sequence_dir = os.path.join(train_dir, sequence)
    os.makedirs(train_sequence_dir, exist_ok=True)
    os.makedirs(os.path.join(train_sequence_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(train_sequence_dir, 'img1'), exist_ok=True)

    shutil.copy(os.path.join(sequence_dir, 'gt', 'gt.txt'), os.path.join(train_sequence_dir, 'gt'))
    shutil.copy(os.path.join(sequence_dir, 'seqinfo.ini'), train_sequence_dir)
    for i, frame in enumerate(frames[:split_index]):
        shutil.copy(os.path.join(images_dir, frame), os.path.join(train_sequence_dir, 'img1'))
        # Update the frame index in the gt.txt file
        # Create a new gt.txt file with updated frame indices
        with open(os.path.join(train_sequence_dir, 'gt', 'gt.txt'), 'r+') as f:
            lines = f.readlines()
            updated_lines = [line.replace(str(i + 1), str(i + 1 - split_index)) for line in lines]
            f.seek(0)
            f.writelines(updated_lines)
            f.truncate()
    # Update the seq length in the seqinfo.ini file
    with open(os.path.join(train_sequence_dir, 'seqinfo.ini'), 'r+') as f:
        lines = f.readlines()
        updated_lines = [line.replace('seqLength = {}'.format(num_frames), 'seqLength = {}'.format(split_index)) for line in lines]
        f.seek(0)
        f.writelines(updated_lines)
        f.truncate()

    # Move the frames to the val directory
    val_sequence_dir = os.path.join(val_dir, sequence)
    os.makedirs(val_sequence_dir, exist_ok=True)
    os.makedirs(os.path.join(val_sequence_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(val_sequence_dir, 'img1'), exist_ok=True)
    shutil.copy(os.path.join(sequence_dir, 'gt', 'gt.txt'), os.path.join(val_sequence_dir, 'gt'))
    shutil.copy(os.path.join(sequence_dir, 'seqinfo.ini'), val_sequence_dir)
    for i, frame in enumerate(frames[split_index:]):
        shutil.copy(os.path.join(images_dir, frame), os.path.join(val_sequence_dir, 'img1'))
        # Update the frame index in the gt.txt file
        with open(os.path.join(val_sequence_dir, 'gt', 'gt.txt'), 'r+') as f:
            lines = f.readlines()
            updated_lines = [line.replace(str(i + 1), str(i + 1 - split_index)) for line in lines]
            f.seek(0)
            f.writelines(updated_lines)
            f.truncate()
        # Update the seq length in the seqinfo.ini file
        with open(os.path.join(val_sequence_dir, 'seqinfo.ini'), 'r+') as f:
            lines = f.readlines()
            updated_lines = [line.replace('seqLength = {}'.format(num_frames), 'seqLength = {}'.format(num_frames - split_index)) for line in lines]
            f.seek(0)
            f.writelines(updated_lines)
            f.truncate()

print('Dataset split and gt.txt update `completed.')