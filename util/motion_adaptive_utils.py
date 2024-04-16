import torch

def scale_bbox(bbox, img_size):
    """
    Scale bbox from [0, 1] to [0, img_size]
    """
    bbox = bbox.clone()
    bbox[0::2] *= img_size[1]
    bbox[1::2] *= img_size[0]
    bbox = bbox.int()
    # Clamp bbox to image size
    bbox[0::2].clamp_(min=0, max=img_size[1])
    bbox[1::2].clamp_(min=0, max=img_size[0])
    return bbox

def motion_adaptive_by_cosine_similarity(cur_flow, cur_bbox, pred_motion, pred_std, weight=0.5):
    """Motion adaptive by cosine similarity between average flow in bbox and predicted motion.

    Args:
        cur_flow: Optical flow of current frame to next frame. Shape (H, W, 2)
        cur_bbox: Current bounding box. Shape (4,)
        pred_motion: Predicted motion. Shape (2,)
        pred_std: Predicted standard deviation . Shape (4,)
        weight: Weight for cosine similarity. Default 0.5.

    Returns:
        adaptive_weight: Adaptive weight based on cosine similarity. Shape (4,)
    """
    cur_bbox = scale_bbox(cur_bbox, cur_flow.shape[:2])
    cur_bbox = cur_bbox.int()

    # Get average flow in bbox
    bbox_flow = cur_flow[cur_bbox[1]:cur_bbox[3], cur_bbox[0]:cur_bbox[2]]
    avg_flow = bbox_flow.mean(dim=(0, 1))

    # Normalize flow
    avg_flow /= avg_flow.norm()
    pred_motion /= pred_motion.norm()

    # Cosine similarity
    cos_sim = avg_flow.dot(pred_motion)
    
    # Compute adaptive weight based on cosine similarity
    adaptive_weight = torch.tensor([-cos_sim * weight, -cos_sim * weight, -pred_std[2], -pred_std[3]]).to(pred_std.device)
    return adaptive_weight


def motion_adaptive_by_standard_deviation(pred_motion, pred_std):
    """Motion adaptive by standard deviation of predicted motion.

    Args:
        pred_motion: Predicted motion. Shape (2,)
        pred_std: Predicted standard deviation . Shape (4,)

    Returns:
        adaptive_weight: Adaptive weight based on standard deviation. Shape (4,)
    """
    adaptive_weight = torch.tensor([-pred_std[0], -pred_std[1], -pred_std[2], -pred_std[3]]).to(pred_std.device)
    return adaptive_weight

def motion_adaptive_by_matched_motion_ratio(cur_flow, cur_bbox, pred_motion, pred_std):
    """ Motion adaptive by matched motion ratio between flow in bbox and predicted motion.

    Args:
        cur_flow: Optical flow of current frame to next frame. Shape (H, W, 2)
        cur_bbox: Current bounding box. Shape (4,)
        pred_motion: Predicted motion. Shape (2,)
        pred_std: Predicted standard deviation . Shape (4,)

    Returns:
        adaptive_weight: Adaptive weight based on matched motion ratio. Shape (4,)
    """
    cur_bbox = scale_bbox(cur_bbox, cur_flow.shape[:2])
    # Get average flow in bbox
    bbox_flow = cur_flow[cur_bbox[1]:cur_bbox[3], cur_bbox[0]:cur_bbox[2]]
    masked_moving_flow = bbox_flow[bbox_flow.norm(dim=-1) > 0.1]
    moving_flow = masked_moving_flow / masked_moving_flow.norm(dim=-1, keepdim=True)
    if (len(moving_flow) == 0):
        # If no motion in bbox, return default weight
        return torch.tensor([-1., -1., -1., -1.]).to(pred_std.device)
    pred_motion = pred_motion / pred_motion.norm()
    # Convert to float
    pred_motion = pred_motion.float()
    matched_ratio = torch.sum((moving_flow @ pred_motion.T) > 0.5) / len(moving_flow) 
    matched_ratio -= 0.5
    # Compute adaptive weight based on matched motion ratio
    adaptive_weight = torch.tensor([matched_ratio, matched_ratio, -pred_std[2], -pred_std[3]]).to(pred_std.device)
    return adaptive_weight