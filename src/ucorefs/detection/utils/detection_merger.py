"""
Detection Merger Utility.
Handles Non-Maximum Suppression (NMS) for merging duplicate detections.
"""
from typing import List, Dict, Optional

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Box format: {'x': float, 'y': float, 'w': float, 'h': float} (normalized or absolute)
    """
    x1_min = box1['x']
    y1_min = box1['y']
    x1_max = box1['x'] + box1['w']
    y1_max = box1['y'] + box1['h']

    x2_min = box2['x']
    y2_min = box2['y']
    x2_max = box2['x'] + box2['w']
    y2_max = box2['y'] + box2['h']

    # Intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    
    inter_area = inter_w * inter_h
    
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area

def merge_detections(detections: List[Dict], iou_threshold: float = 0.9) -> List[Dict]:
    """
    Merge overlapping detections using Non-Maximum Suppression (NMS).
    
    Criteria:
    1. Sort by confidence (descending).
    2. If IoU > threshold AND label is same -> Suppress lower confidence one.
    
    Args:
        detections: List of dicts with 'bbox', 'label', 'confidence'
        iou_threshold: Threshold for merging (default 0.9)
        
    Returns:
        List of merged (kept) detections.
    """
    if not detections:
        return []
        
    # Validations to prevent crashes on bad data
    valid_dets = []
    for d in detections:
        if 'bbox' in d and 'label' in d:
             valid_dets.append(d)
    
    # Sort by confidence desc
    sorted_dets = sorted(valid_dets, key=lambda x: x.get('confidence', 0.0), reverse=True)
    kept = []
    
    while sorted_dets:
        current = sorted_dets.pop(0)
        kept.append(current)
        
        # Compare minimal set of remaining detections
        remaining = []
        for other in sorted_dets:
            should_merge = False
            
            # Label check (must be same to merge)
            if other['label'] == current['label']:
                iou = calculate_iou(current['bbox'], other['bbox'])
                if iou >= iou_threshold:
                    should_merge = True
            
            if not should_merge:
                remaining.append(other)
        
        sorted_dets = remaining
        
    return kept
