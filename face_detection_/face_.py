import cv2
from PIL import Image
import dlib
import os

def face_detection_opencv(img, file = 'haarcascade_frontalface_default.xml', x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550, cuda=True):
    image = []
    img_mask = img[x1_mask:x2_mask, y1_mask:y2_mask]
    if cuda:
        cass_path = "../haarcascades_cuda"
    else:
        cass_path = "../haarcascades"
    
    cass_path = os.path.join(cass_path, file)
    face_cascade = cv2.CascadeClassifier(cass_path)
    gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        
        print(x, y, w, h)
        x1 = x + x1_mask
        y1 = y + y1_mask
        x2 = x1 + h
        y2 = y1 + w
        p = [x1, y1, x2, y2]
        image.append(p)
        
    
    return image

def face_detection_dlib(img, x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550):
    # 构建人脸检测器
    detector = dlib.get_frontal_face_detector()
    
    image = []
    img_mask = img[x1_mask:x2_mask, y1_mask:y2_mask]
    
    # cv2.imshow('aa', img_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    b, g, r = cv2.split(img_mask)
    img_mask = cv2.merge([r, g, b])
    dets = detector(img_mask, 1)
    for index, face in enumerate(dets):
        # 人脸位置
        left = face.left() + y1_mask
        top = face.top() + x1_mask
        right = face.right() + y1_mask
        bottom = face.bottom() + x1_mask
        weizhi = [left, top, right, bottom]
        image.append(weizhi)

    return image, dets


    



