import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import datasets, models, transforms
from . import model_vgg

def shibie(image):

    model = model_vgg.VGG()

    model.load_state_dict(torch.load('../model_/params.pkl'))

    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = data_transform(img)
    img = img.view(1, 3, 48, 48)

    output = model(img)
    _, preds = torch.max(output, 1)
    preds = preds[0].item()
    expression_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 
                        4: 'happy', 5: 'sadness', 6: 'surprise'}

    expression_state = "expression:" + expression_dict[preds]

    return expression_state

# if __name__ == "__main__":
#     file_path = '/home/wang/b.jpg'
#     img = cv2.imread(file_path)
#     text = shibie(img)
#     print(text)