"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
import torch
import copy

from models.structures.instances import Instances
from util import box_ops
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
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

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
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
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
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}

class MotionPrediction(object):
    def __init__(self, args):
        self.delta_t = args.delta_t
        self.is_origin_kalman = args.is_origin_kalman
        self.weight_prev_wh = args.weight_prev_wh
        self.weight_prev_xy = args.weight_prev_xy
        self.is_adaptive_by_std = args.is_adaptive_by_std
    def _update_bbox(self, track_instances: Instances):
        device = track_instances.pred_boxes.device
        is_none_tracker = torch.tensor([tracker is None for tracker in track_instances.tracker]).to(device)
        pred_bboxes = track_instances.pred_boxes.detach().clone()
        # Convert to xyxy
        pred_bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes)
        # Update trackers
        for i, tracker in enumerate(track_instances.tracker):
            if tracker is not None:
                if (track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] > 0.5):
                    # print("Object index:", track_instances.obj_idxes[i])
                    tracker.update(pred_bboxes[i].cpu().numpy())
                else:
                    tracker.update(None)

                predicted_boxes, std = tracker.predict()

                if (track_instances.scores[i] >= 0.5 and tracker.hit_streak > 5):
                    predicted_boxes = torch.tensor(predicted_boxes).to(device)
                    # print("History:", tracker.history_observations[-3:])
                    # print("Predicted boxes:", predicted_boxes)
                    # print("Std:", std)
                    # Convert to cxcywh
                    predicted_boxes = box_ops.box_xyxy_to_cxcywh(predicted_boxes).squeeze(0)
                    # Get diag of std as uncertainty and only use the first 2 dimensions (x, y)
                    # # print(std)
                    # # print(tracker.hits, tracker.hit_streak, tracker.age, tracker.time_since_update)
                    std_xy = torch.tensor(std[:2]).to(device).clamp(0, 1)
                    std_r = torch.tensor(std[3:4]).to(device).clamp(0, 1)
                    # # print(std_xy, std_r)
                    std_xywh = torch.cat([std_xy, std_r, std_r]).to(device)
                    # # Only update the center of the box
                    # # print(track_instances.obj_idxes[i])
                    # # print(std_xywh)
                    pred_xy_weight = 1.0 - self.weight_prev_xy
                    pred_wh_wieght = 1.0 - self.weight_prev_wh
                    prev_xy_weight = (self.weight_prev_xy + std_xywh[:2] * pred_xy_weight * self.is_adaptive_by_std)
                    prev_wh_weight = (self.weight_prev_wh + std_xywh[2:] * pred_wh_wieght * self.is_adaptive_by_std)
                    predicted_boxes[:2] = track_instances.pred_boxes[i][:2] * prev_xy_weight + \
                                            predicted_boxes[:2] * (1.0 - prev_xy_weight)
                    predicted_boxes[2:] = track_instances.pred_boxes[i][2:] * prev_wh_weight + \
                                            predicted_boxes[2:] * (1.0 - prev_wh_weight)
                    track_instances.ref_pts[i] = predicted_boxes.clamp(0, 1)
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

    def __call__(self, track_instances: Instances) -> Instances:
        track_instances = self._update_bbox(track_instances)
        return track_instances
    
def build(args):
    return MotionPrediction(args)