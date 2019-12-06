import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

def data_set_train(BATCH_SIZE, DATA_FILE_ROOT):
  
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    expression_dataset = datasets.ImageFolder(root = DATA_FILE_ROOT, 
                                            transform = data_transform)

    dataset_loader = torch.utils.data.DataLoader(expression_dataset,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                num_workers = 4)

    return dataset_loader


def data_set_val(BATCH_SIZE, DATA_FILE_ROOT):
    
    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    expression_dataset = datasets.ImageFolder(root = DATA_FILE_ROOT, 
                                            transform = data_transform)

    dataset_loader = torch.utils.data.DataLoader(expression_dataset,
                                                batch_size = BATCH_SIZE,
                                                shuffle = False,
                                                num_workers = 4)

    return dataset_loader