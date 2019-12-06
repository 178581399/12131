import cv2
import os
import sys
# sys.path.append('../')
import test
from tqdm import tqdm

def add_img(file_root, file_save, n = 5):
    
    file_name = os.listdir(file_root)
    # file_name = os.path.join(file_root, file_name)
    for i, path in enumerate(tqdm(file_name)):
        file_name_all = os.path.join(file_root, path)

        if i % n == 0:
            text = test.shibie(file_name_all)
        
        img = cv2.imread(file_name_all)
        cv2.putText(img, text, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255), 2)
        i = str(i)
        svae_path = file_save + i + ".jpg"
        cv2.imwrite(svae_path, img)

if __name__ == "__main__":
    file_root = '/home/wang/gongzuo/img'
    file_save = '/home/wang/gongzuo/jieshu/'
    n = 5
    add_img(file_root, file_save, n)


