import torch
import torchvision
from torch import nn
from models.structures import Instances

from util import box_ops


class MemoryBankBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self.max_his_length = args.memory_bank_len
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, track_instances: Instances) -> Instances:
        raise NotImplementedError()

    def _update_memory_bank(self, track_instances):
        raise NotImplementedError()


class SparsenessAwareMemoryModule(MemoryBankBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        # Scale to 0-1 relative to image size
        self.max_accum_dist = args.memory_bank_max_dist
        self.max_frame_gap = args.memory_bank_max_frame_gap
        self.iou_threshold = args.memory_bank_iou_threshold
        self.high_conf_threshold = args.memory_bank_high_conf_threshold
        self.temporal_gap_reg = args.memory_bank_temporal_gap_regulization

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        pass

    def _reset_track_instances(self, track_instances, is_update):
        track_instances.accum_dist[is_update] = 0
        track_instances.best_iou[is_update] = 1
        track_instances.best_features[is_update] = torch.zeros_like(track_instances.best_features[is_update])
    
    def _update_best_features_by_iou(self, track_instances):
        pred_boxes = track_instances.pred_boxes.detach().clone()
        # Increase the best_iou as regularization of temporal gap
        track_instances.best_iou = track_instances.best_iou.detach().clone() * (1 + self.temporal_gap_reg)
        # Convert to xyxy format
        pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)

        ious = torchvision.ops.box_iou(pred_boxes, pred_boxes)

        # Remove self iou and masked ious
        ious[range(len(ious)), range(len(ious))] = 0
        ious = ious.max(dim=-1).values

        is_update_best_features = ious <= track_instances.best_iou
        track_instances.best_iou[is_update_best_features] = ious[is_update_best_features]
        track_instances.best_features[is_update_best_features] = track_instances.query_pos[is_update_best_features].detach().clone()

        return track_instances

    def _check_is_update(self, track_instances):
        frame_gaps = track_instances.frame_idx - track_instances.mem_frames_idx[:, -1]
        accum_dists = track_instances.accum_dist
        is_pos = track_instances.scores > self.high_conf_threshold
        
        is_update = (accum_dists > self.max_accum_dist)
        is_update = is_update & is_pos
        is_update = is_update | (frame_gaps > self.max_frame_gap)

        return is_update

    def _update_memory_bank(self, track_instances):
        """Update memory bank based on accumulated moving distance of objects. 
        Args:
            track_instances (Instances): Instances of tracks

        Returns:
            Instances: Updated instances of tracks
        """
        if len(track_instances) == 0:
            return track_instances

        track_instances = self._update_best_features_by_iou(track_instances)

        is_update = self._check_is_update(track_instances)
        
        new_mem_frames_idx = track_instances.mem_frames_idx[is_update].clone()
        new_mem_bank = track_instances.mem_bank[is_update].clone()
        new_mem_padding_mask = track_instances.mem_padding_mask[is_update].clone()
        # Shifted all mem_bank and mem_padding_mask of updated idx to left
        new_mem_frames_idx[:, :-1] = track_instances.mem_frames_idx[is_update][:, 1:]
        new_mem_bank[:, :-1] = track_instances.mem_bank[is_update][:, 1:]
        new_mem_padding_mask[:, :-1] = track_instances.mem_padding_mask[is_update][:, 1:]
        # Update the last column of mem_bank and mem_padding_mask
        new_mem_frames_idx[:, -1] = track_instances.frame_idx[is_update]
        new_mem_bank[:, -1] = track_instances.best_features[is_update].detach().clone()
        new_mem_padding_mask[:, -1] = torch.zeros_like(new_mem_padding_mask[:, -1])
        
        track_instances.mem_frames_idx[is_update] = new_mem_frames_idx
        track_instances.mem_bank[is_update] = new_mem_bank
        track_instances.mem_padding_mask[is_update] = new_mem_padding_mask

        # Reset updated track
        self._reset_track_instances(track_instances, is_update)

        return track_instances

    def forward(self, track_instances: Instances) -> Instances:
        # track_instances = self._select_active_tracks(track_instances)
        track_instances = self._update_memory_bank(track_instances)
        return track_instances


def build(args, dim_in, hidden_dim, dim_out):
    memory_banks = {
        'sam': SparsenessAwareMemoryModule,
    }
    return memory_banks[args.memory_bank_type](args, dim_in, hidden_dim, dim_out)
