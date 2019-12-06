import sys
import os
import dlib
import cv2


def face_landmarks(img, weizhi, x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550):
    # 加载人脸关键点检测模型
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    predictor_path = "../face_land_/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # print(weizhi.left())
    # left = weizhi.left() + y1_mask
    # weizhi.left() = left
    # weizhi.left() += y1_mask
    # weizhi.top() += x1_mask
    # weizhi.right() += y1_mask
    # weizhi.bottom() += x1_mask
    # print(weizhi.left())
    left = weizhi.left() + y1_mask
    top = weizhi.top() + x1_mask
    right = weizhi.right() + y1_mask
    bottom = weizhi.bottom() + x1_mask
    # print(type(left))
    # print(type(top))
    # print(type(right))
    # print(type(bottom))

    # imga = img[top:bottom,left:right]
    # cv2.imshow('a', imga)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    wz = dlib.rectangle(left, top, right, bottom)
    shape = predictor(img, wz)
    
    return shape
