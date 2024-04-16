"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import copy

import numpy as np
import torch
import torchvision

from models.structures.instances import Instances
from util import box_ops
from util import motion_adaptive_utils

from .association import *


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def motion_similarity(cur_flow, cur_bbox, prev_motion):
    """
    Optical flow consistency estimation.
    Args:
        cur_flow: optical flow from previous frame to current frame
        cur_bbox: bounding box of the object in the current frame
        prev_motion: previous motion vector
    Returns:
        cos_sim: cosine similarity between the average flow and the previous motion
    """
    scale_x = cur_flow.shape[1]
    scale_y = cur_flow.shape[0] 
    scale_box = np.array([scale_x, scale_y, scale_x, scale_y])
    
    scaled_bbox = (cur_bbox * scale_box).astype(int)
    # Clamp cur bbox to the image size
    scaled_bbox[0] = max(0, scaled_bbox[0])
    scaled_bbox[1] = max(0, scaled_bbox[1])
    scaled_bbox[2] = min(cur_flow.shape[1], scaled_bbox[2])
    scaled_bbox[3] = min(cur_flow.shape[0], scaled_bbox[3])

    mask = np.zeros_like(cur_flow[:, :, 0], dtype=bool)
    mask[scaled_bbox[1]:scaled_bbox[3], scaled_bbox[0]:scaled_bbox[2]] = 1
    masked_avg_flow = np.mean(cur_flow[mask], axis=0)

    # Compute cosine similarity between the average flow and the previous motion
    cos_sim = np.dot(masked_avg_flow, prev_motion) / \
        (np.linalg.norm(masked_avg_flow) * np.linalg.norm(prev_motion))
    return cos_sim

def motion_matched_ratio(cur_flow, cur_bbox, prev_motion):
    """Compute ratio of matched motion vectors in the bounding box with the previous motion vector"""
    scale_x = cur_flow.shape[1]
    scale_y = cur_flow.shape[0] 
    scale_box = np.array([scale_x, scale_y, scale_x, scale_y])
    
    scaled_bbox = (cur_bbox * scale_box).astype(int)
    # Clamp cur bbox to the image size
    scaled_bbox[0] = max(0, scaled_bbox[0])
    scaled_bbox[1] = max(0, scaled_bbox[1])
    scaled_bbox[2] = min(cur_flow.shape[1], scaled_bbox[2])
    scaled_bbox[3] = min(cur_flow.shape[0], scaled_bbox[3])

    mask = np.zeros_like(cur_flow[:, :, 0], dtype=bool)
    mask[scaled_bbox[1]:scaled_bbox[3], scaled_bbox[0]:scaled_bbox[2]] = 1
    masked_flow = cur_flow[mask]
    # Moving flow masked
    moving_flow = masked_flow[np.linalg.norm(masked_flow, axis=1) > 0.1]
    # Compute ratio of matched motion vectors
    matched_ratio = np.sum(np.dot(moving_flow, prev_motion) > 0.5) / len(moving_flow)
    return matched_ratio

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, dim_x=7):
        """
        Initialises a tracker using initial bounding box.

        """
        self.bbox_scale = 100.0
        dim_z = 4
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
            self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        # self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
        #                     0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
        #                     [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.F = np.zeros((dim_x, dim_x))
        # Set the state transition matrix values with 1 in the diagonal and 1 in the velocity columns
        for i in range(dim_x):
            self.kf.F[i, i] = 1
            if i+dim_z < dim_x:
                self.kf.F[i, i+dim_z] = 1

        self.kf.H = np.zeros((4, dim_x))
        # Set the observation matrix values with 1 in the diagonal
        for i in range(4):
            self.kf.H[i, i] = 1

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            bbox *= self.bbox_scale
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # Get uncertainty of the prediction
        std = np.sqrt(np.diag(self.kf.P))
        return self.history[-1] / self.bbox_scale, std / self.bbox_scale

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {"iou": iou_batch,
              "giou": giou_batch,
              "ciou": ciou_batch,
              "diou": diou_batch,
              "ct_dist": ct_dist}

ADAPTIVE_FUNCS = {"motion_similarity": motion_adaptive_utils.motion_adaptive_by_cosine_similarity,
                  "motion_matched_ratio": motion_adaptive_utils.motion_adaptive_by_matched_motion_ratio,
                  "standard_deviation": motion_adaptive_utils.motion_adaptive_by_standard_deviation}


class MotionPrediction(object):
    def __init__(self, args):
        self.delta_t = args.delta_t
        self.is_origin_kalman = args.is_origin_kalman
        self.weight_prev_wh = args.weight_prev_wh
        self.weight_prev_xy = args.weight_prev_xy
        self.is_adaptive_by_std = args.is_adaptive_by_std
        self.is_adaptive_by_flow = args.is_adaptive_by_flow
        self.motion_adaptive_func = args.motion_adaptive_func

        # Backward compatibility
        if (self.is_adaptive_by_flow):
            self.motion_adaptive_func = "motion_similarity"
        if (self.is_adaptive_by_std):
            self.motion_adaptive_func = "standard_deviation"

        self.adaptive_func = ADAPTIVE_FUNCS[self.motion_adaptive_func]

    def _update_bbox(self, track_instances: Instances, flow):
        device = track_instances.pred_boxes.device
        is_none_tracker = torch.tensor(
            [tracker is None for tracker in track_instances.tracker]).to(device)
        pred_bboxes = track_instances.pred_boxes.detach().clone()
        # Convert to xyxy
        pred_bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes)
        iou_mask = torchvision.ops.box_iou(
            pred_bboxes, pred_bboxes).to(device)
        
        # Remove self iou
        iou_mask = iou_mask - torch.eye(iou_mask.size(0)).to(device)
        iou_mask = iou_mask.max(dim=1)[0]
        # Update trackers
        for i, tracker in enumerate(track_instances.tracker):
            if tracker is not None:
                if (track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] > 0.5):
                    # print("Object index:", track_instances.obj_idxes[i])
                    tracker.update(pred_bboxes[i].cpu().numpy())
                else:
                    tracker.update(None)

                kf_pred_boxes, std = tracker.predict()

                if (track_instances.scores[i] >= 0.5 and tracker.hit_streak > 5):
                    kf_pred_boxes = torch.tensor(kf_pred_boxes).to(device)
                    kf_pred_boxes = box_ops.box_xyxy_to_cxcywh(
                        kf_pred_boxes).squeeze(0)

                    pred_xy_weight = 1.0 - self.weight_prev_xy
                    pred_wh_weight = 1.0 - self.weight_prev_wh
                    std_xy = torch.tensor(std[:2]).to(device).clamp(0, 1)
                    std_r = torch.tensor(std[3:4]).to(device).clamp(0, 1)
                    std_xywh = torch.cat([std_xy, std_r, std_r]).to(device)
                    # if (self.is_adaptive_by_std or iou_mask[i] > 0.5):
                    #     # If the uncertainty is high, we trust the previous prediction
                    #     prev_xy_weight = (
                    #         self.weight_prev_xy + std_xywh[:2] * pred_xy_weight)
                    #     prev_wh_weight = (
                    #         self.weight_prev_wh + std_xywh[2:] * pred_wh_weight)
                    # elif (self.is_adaptive_by_flow):
                    #     motion_sim = motion_similarity(
                    #         flow, pred_bboxes[i].cpu().numpy(), 
                    #         (kf_pred_boxes.cpu().numpy() - pred_bboxes[i].cpu().numpy())[:2])
                    #     # Shift motion sim to 0.5
                    #     motion_displacement = -motion_sim * 0.5
                    #     # If the motion is consistent with the previous motion, we trust the prediction of the tracker
                    #     prev_xy_weight = self.weight_prev_xy + motion_displacement * pred_xy_weight
                    #     # Motion can't be used to predict the aspect ratio
                    #     prev_wh_weight = (
                    #         self.weight_prev_wh + std_xywh[2:] * pred_wh_weight)
                    # elif (self.is_adaptive_)
                    # else:
                    #     prev_xy_weight = self.weight_prev_xy
                    #     prev_wh_weight = self.weight_prev_wh
                    if (self.motion_adaptive_func == "standard_deviation"):
                        adaptive_weight = self.adaptive_func(std_xywh, std_xywh)
                    else:
                        adaptive_weight = self.adaptive_func(
                            flow, pred_bboxes[i], 
                            (kf_pred_boxes - pred_bboxes[i])[:2], std_xywh)
                        
                    prev_xy_weight = self.weight_prev_xy - adaptive_weight[:2] * pred_xy_weight
                    prev_wh_weight = self.weight_prev_wh - adaptive_weight[2:] * pred_wh_weight

                    kf_pred_boxes[:2] = track_instances.pred_boxes[i][:2] * prev_xy_weight + \
                        kf_pred_boxes[:2] * (1.0 - prev_xy_weight)
                    kf_pred_boxes[2:] = track_instances.pred_boxes[i][2:] * prev_wh_weight + \
                        kf_pred_boxes[2:] * (1.0 - prev_wh_weight)

                    track_instances.ref_pts[i] = kf_pred_boxes.clamp(0, 1)
                    print("tracker:", i, "pred:", kf_pred_boxes)
                    # print("tracker:", i, "pred:", kf_pred_boxes)
                    # print("-------------------------------")

        # Check if trackers are not initialized
        new_trackers = [KalmanBoxTracker(bbox.cpu().numpy(), delta_t=self.delta_t,
                                         orig=self.is_origin_kalman,
                                         dim_x=7)
                        for bbox in pred_bboxes[is_none_tracker]]
        # Update trackers
        for i, tracker in enumerate(track_instances.tracker):
            if tracker is None:
                track_instances.tracker[i] = new_trackers.pop(0)
                track_instances.tracker[i].predict()
        return track_instances

    def __call__(self, track_instances: Instances, flow) -> Instances:
        track_instances = self._update_bbox(track_instances, flow)
        return track_instances


def build(args):
    return MotionPrediction(args)
