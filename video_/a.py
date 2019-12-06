import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append("../")
from face_detection_ import face_
from model_ import test_
from face_land_ import face_landmarks

class video_res():
    # x1, x2, y1, y2: 原图像的裁剪范围
    # x, y: 图像增加的大小
    # x_mask: 拼接之后的图片提取人脸的mask范围
    # cuda: 检测是否使用GPU
    # file: 所所使用的file名字
    # oc 是否使用opencv 否则使用dlib进行检测
    def __init__(self, video_path, cuda = True, oc = False, x1 = 317, x2 = 1680, y1 = 25, y2 = 2800, x = 1363, y = 500):
        self.video_path = video_path 
        self.cuda = cuda
        self.oc = oc
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.x = x
        self.y = y
        
    # n: 每n帧处理一次
    def video_r(self, file = 'haarcascade_frontalface_default.xml', x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550, n = 1):
        cap = cv2.VideoCapture(self.video_path)
        # 从指定的帧数开始读取视频
        cap.set(cv2.CAP_PROP_POS_FRAMES,4343)
        dim = np.zeros((self.x, self.y, 3))
        dim[:, :, :] = 255
        ii = 0
        # 返回视频中人脸的位置
        frame_mask = []
        # 在视频中所要显示的心情
        expression_text = ""
        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                break
            if ii % n == 0:
                frame = frame[self.x1:self.x2, self.y1:self.y2]
                frame_p = np.concatenate((dim, frame), axis = 1)
                frame_p = frame_p.astype(np.uint8)

                if self.oc:
                    print("无法进行人脸关键点检测,请将参数oc设为False !!!")
                    exit()
                    frame_mask = face_.face_detection_opencv(frame_p, file, x1_mask, x2_mask, y1_mask, y2_mask, self.cuda)
                else:
                    frame_mask, dets = face_.face_detection_dlib(frame_p, x1_mask, x2_mask, y1_mask, y2_mask)
                

                for i, weizhi in enumerate(frame_mask): 
                    x1, y1, x2, y2 = weizhi 
                    # print(type(dets[i]))
                    # 人脸关键点检测
                    shape = face_landmarks.face_landmarks(frame_p, dets[i])
                    # print(type(shape))
                    face_p = frame_p[y1:y2, x1:x2]
                    # cv2.imwrite('/home/wang/b.jpg', face_p)
                    expression_text = test_.shibie(face_p)
                    break
                    
            text = self.chongzu(face_p, expression_text, shape, x1, y1, x2, y2)
            self.drawing(frame_p, text, shape, x1, y1, x2, y2, ii, x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550)
            if ii % 10 == 0:
                print(ii)
            ii += 1

    # frame_mask: 人脸的位置
    # expression_text: 表情信息
    def chongzu(self, face_p, expression_text, shape, *weizhi):
        x1, y1, x2, y2 = weizhi
        text = "{" + "\n"
        text += ("  %s\n") % expression_text
        text += ("  'face_land:' %d, %d, %d, %d \n") % (x1, y1, x2, y2)
        # 人脸关键点:
        text += "  'face':{\n"
        # 人脸外框 0
        text += "    'profile_left:'{\n"
        text += ("    'x:'%d, \n") % shape.part(0).x
        text += ("    'y:'%d, \n") % shape.part(0).y
        text += "    },"
        # 人脸外框 8 下巴
        text += "    'chin:'{\n"
        text += ("    'x:'%d, \n") % shape.part(8).x
        text += ("    'y:'%d, \n") % shape.part(8).y
        text += "    },"
        # 人脸外框 15
        text += "    'profile_right:'{\n"
        text += ("    'x:'%d, \n") % shape.part(15).x
        text += ("    'y:'%d, \n") % shape.part(15).y
        text += "    },"
        # 眉毛 left 19
        text += "    'left_eyebrow:'{\n"
        text += ("    'x:'%d, \n") % shape.part(19).x
        text += ("    'y:'%d, \n") % shape.part(19).y
        text += "    },"
        # 眉毛 right 24
        text += "    'right_eyebrow:'{\n"
        text += ("    'x:'%d, \n") % shape.part(24).x
        text += ("    'y:'%d, \n") % shape.part(24).y
        text += "    },"
        # 左眼 37
        text += "    'left_eye:'{\n"
        text += ("    'x:'%d, \n") % shape.part(37).x
        text += ("    'y:'%d, \n") % shape.part(37).y
        text += "    },"
        # 右眼 43
        text += "    'right_eye:'{\n"
        text += ("    'x:'%d, \n") % shape.part(43).x
        text += ("    'y:'%d, \n") % shape.part(43).y
        text += "    },"
        # 鼻子 30
        text += "    'nose:'{\n"
        text += ("    'x:'%d, \n") % shape.part(30).x
        text += ("    'y:'%d, \n") % shape.part(30).y
        text += "    },"
        # 嘴巴 左嘴角
        text += "    'mouth_left:'{\n"
        text += ("    'x:'%d, \n") % shape.part(48).x
        text += ("    'y:'%d, \n") % shape.part(48).y
        text += "    },"
        # 嘴巴 右嘴角
        text += "    'mouth_right:'{\n"
        text += ("    'x:'%d, \n") % shape.part(50).x
        text += ("    t'y:'%d, \n") % shape.part(50).y
        text += "    }, \n"
        # 结束
        text += "}"

        return text

    def drawing(self, img, text, shape, x1, y1, x2, y2, ii, x1_mask = 100, x2_mask = 950, y1_mask = 1650, y2_mask = 2550):
        # x1, y1, x2, y2 = weizhi
        # y0 起始行数
        # dy 每一行的间距
        y0 = 40
        dy = 36
        for i, txt in enumerate(text.split('\n')):
            y = y0+i*dy
            cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for index in range(68):
            x = shape.part(index).x
            y = shape.part(index).y
            cv2.circle(img, (x, y), 1, (0, 0, 255), 6)

        file_path = ('../images/%d.jpg') % ii
        cv2.imwrite(file_path, img)
        return 
        # cv2.imwrite('/home/wang/z.jpg', img)
        # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        # cv2.imshow('af', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        
if __name__ == "__main__":
    
    j = video_res('/home/wang/gongzuo/xuanxuan111.mp4', oc = False)
    j.video_r()