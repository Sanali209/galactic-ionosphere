import pytest
from src.ucorefs.detection.utils.detection_merger import merge_detections, calculate_iou

def test_iou_calculation():
    # Identical boxes
    b1 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    b2 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    assert calculate_iou(b1, b2) == 1.0
    
    # No overlap
    b3 = {'x': 20, 'y': 20, 'w': 10, 'h': 10}
    assert calculate_iou(b1, b3) == 0.0
    
    # 50% overlap (approx) - actually let's calculate exact
    # b1: 0,0 to 10,10. b4: 5,0 to 15,10.
    # Inter: 5,0 to 10,10 -> w=5, h=10 -> area=50.
    # Union: 100 + 100 - 50 = 150.
    # IoU = 50/150 = 0.333
    b4 = {'x': 5, 'y': 0, 'w': 10, 'h': 10}
    assert abs(calculate_iou(b1, b4) - 0.333333) < 0.001

def test_merge_exact_duplicates():
    dets = [
        {"label": "person", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}, "confidence": 0.9},
        {"label": "person", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}, "confidence": 0.8}
    ]
    merged = merge_detections(dets, iou_threshold=0.9)
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.9

def test_merge_different_labels_dont_merge():
    # Identical boxes but different labels
    dets = [
        {"label": "person", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}, "confidence": 0.9},
        {"label": "statue", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}, "confidence": 0.8}
    ]
    merged = merge_detections(dets, iou_threshold=0.9)
    assert len(merged) == 2

def test_nms_threshold_behavior():
    # IoU approx 0.33 (from above), threshold 0.5 -> Should NOT merge
    dets = [
        {"label": "person", "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "confidence": 0.9},
        {"label": "person", "bbox": {"x": 5, "y": 0, "w": 10, "h": 10}, "confidence": 0.8} # IoU 0.33
    ]
    
    # High threshold -> No merge
    merged_high = merge_detections(dets, iou_threshold=0.5)
    assert len(merged_high) == 2
    
    # Low threshold -> Merge
    merged_low = merge_detections(dets, iou_threshold=0.1)
    assert len(merged_low) == 1
    assert merged_low[0]["confidence"] == 0.9

def test_empty_list():
    assert merge_detections([]) == []

def test_missing_fields_ignored():
    dets = [
        {"label": "person", "confidence": 0.9}, # Missing bbox
        {"bbox": {"x": 0, "y": 0}, "confidence": 0.1} # Missing label
    ]
    merged = merge_detections(dets)
    assert len(merged) == 0
