from pathlib import Path
import cv2, numpy as np
from app import preprocess, lane_overlay, iou

def test_preprocess():
    x=np.zeros((360,640,3),dtype=np.uint8); y=preprocess(x); assert y.shape==x.shape

def test_lane_overlay():
    x=np.zeros((360,640,3),dtype=np.uint8); y=lane_overlay(x); assert y.shape==x.shape

def test_iou():
    assert abs(iou([0,0,10,10],[5,5,15,15])-.142857)<1e-4

if __name__=='__main__':
    test_preprocess(); test_lane_overlay(); test_iou(); print('ALL PIPELINE TESTS PASSED')
