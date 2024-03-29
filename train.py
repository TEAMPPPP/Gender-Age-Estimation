import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from torchvision import transforms

from tqdm.notebook import tqdm
import timm
import shutil

base_path = "/home/jhun/Age_gender_estimation/data/UTKFace"
files = os.listdir(base_path)
size = len(files)

class CustomDataset(Dataset):
    def __init__(self, files):
        self.target = []
        for items in files:
            split_var = items.split('_')
            age = int(split_var[0])
            gender = int(split_var[1])
            self.target.append(np.array([self.age_group(age), gender], dtype = object))
        self.files = files
        self.target = np.array(self.target)
    
    def __len__(self):
        return len(self.target)

    #Encode age to 20 parts
    @staticmethod
    def age_group(age):
        group = np.zeros(20)
        idx = 0
        for i in range(0, 100, 5):
            if age >= i and age <= i+4 and i%5 == 0:
                idx = i/5

        group[int(idx)] = 1
        return group

    def __getitem__(self, idx):
        image_path = os.path.join(base_path, self.files[idx])
        image = Image.open(image_path)
        
        # Need to change the image to match Resnet50 input
        T = transforms.Compose([
            transforms.Resize([256], torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop([224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        image = T(image)
        
        age = torch.tensor(self.target[idx][0])
        gender = torch.unsqueeze(torch.tensor(self.target[idx][1], dtype=torch.float), dim=0)
        return image, [age, gender]

    
    def remove(self, idx, criteria):
        if criteria:
            self.files.remove(self.files[idx])
            self.target.remove(self.target[idx])
            
random.seed(42)
np.random.shuffle(files)
train_len = int(round(size/100*80, 0))
test_len = int(round(size/100*20, 0))
train_images, test_images = files[:train_len], files[train_len:]

train_data = CustomDataset(train_images)
test_data = CustomDataset(test_images)
generator1 = torch.Generator().manual_seed(42)

train_size = len(train_data)  # Get the actual length of the train_data dataset
val_size = int(0.2 * train_size)  # Calculate 20% of train_data for validation
train_size = train_size - val_size  # Adjust train_size to be the remainder

# Now, use the calculated train_size and val_size for splitting
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size], generator=generator1)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model in pytorch
class AgeNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        # self.backbone = torchvision.models.swin_v2_b(weights = 'DEFAULT')
        self.backbone = torchvision.models.resnet50(weights = 'DEFAULT')
        # self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        # Define 'necks' for each head
        self.age = nn.Sequential(
            nn.Linear(1000, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
            )
        
    def forward(self, x):
        x = self.backbone(x)
        out_age = self.age(x)
        return out_age
    
# Define model in pytorch
class GenderNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        # self.backbone = torchvision.models.swin_v2_b(weights = 'DEFAULT')
        self.backbone = torchvision.models.resnet50(weights = 'DEFAULT')

        # Define 'necks' for each head
        self.gender = nn.Sequential(
            nn.Linear(1000, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.backbone(x)
        out_gender = self.gender(x)
        return out_gender

def save_checkpoint(state, is_best):
    torch.save(state['age_state_dict'], '/home/jhun/Age_gender_estimation/pretrained/age_model20_checkpoint_split.pth')
    torch.save(state['gender_state_dict'], '/home/jhun/Age_gender_estimation/pretrained/gender_model20_checkpoint_split.pth')

    if is_best:
        print('Found best')
        torch.save(state['age_state_dict'], '/home/jhun/Age_gender_estimation/pretrained/age_model20_checkpoint_best.pth')
        torch.save(state['gender_state_dict'], '/home/jhun/Age_gender_estimation/pretrained/gender_model20_checkpoint_best.pth')
            
age_model = AgeNeuralNetwork().to(device)
gender_model = GenderNeuralNetwork().to(device)

# Define loss
loss_age_fn = nn.CrossEntropyLoss()
loss_gender_fn = nn.BCELoss()

# Optimizer
#import torch.optim.lr_scheduler as lr_scheduler
optimizer_age = torch.optim.Adam(age_model.parameters(), lr=1e-3)
optimizer_gender = torch.optim.Adam(gender_model.parameters(), lr=1e-3)

# define training hyperparameters
BATCH_SIZE = 64
EPOCHS = 20

train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, num_workers=2)
val_dataloader = DataLoader(val_data, batch_size = 10, num_workers=2)

age_train_loss_total, gender_train_loss_total = [], []
age_val_loss_total, gender_val_loss_total = [], []
age_acc_total, gender_acc_total = [], []
best_acc = 0

for epoch in tqdm(range(EPOCHS)):
    # Training loop
    print('Train-----')
    age_epoch_loss, gender_epoch_loss = 0, 0

    age_model.train()
    gender_model.train()
    for batch_data, batch_labels in tqdm(train_dataloader):
        batch_data = batch_data.to(device)
        
        # Forward pass
        age_label = batch_labels[0].to(device)            
        gender_label = batch_labels[1].to(device)           
      
        age_output = age_model(batch_data)
        gender_output = gender_model(batch_data)

        age_loss = loss_age_fn(age_output, age_label)
        age_epoch_loss += age_loss.item()

        gender_loss = loss_gender_fn(gender_output, gender_label)
        gender_epoch_loss += gender_loss.item()

        # Backward pass and update the model parameters
        # Zero the gradients
        age_loss.backward()
        optimizer_age.step()
        optimizer_age.zero_grad()
        
        gender_loss.backward()
        optimizer_gender.step()
        optimizer_gender.zero_grad()


    # Combined loss
    age_epoch_loss /= BATCH_SIZE
    gender_epoch_loss /= BATCH_SIZE

    age_train_loss_total.append(age_epoch_loss)
    gender_train_loss_total.append(gender_epoch_loss)


    print(f"Epoch {epoch}")
    print(f'Loss age:{age_epoch_loss}')
    print(f'Loss gender:{gender_epoch_loss}')


    # Vaidating
    size = len(val_dataloader.dataset)
    num_batches = len(val_dataloader)
    age_model.eval()
    gender_model.eval()
    test_age_loss, test_gender_loss = 0, 0
    correct_age, correct_gender = 0, 0
    with torch.no_grad():
        print('Val------')
        for X, y in tqdm(val_dataloader):
            X = X.to(device)
            
            age_pred = age_model(X)
            gender_pred = gender_model(X)


            age_label = y[0].to(device)            
            gender_label = y[1].to(device)          


            test_age_loss += loss_age_fn(age_pred, age_label).item()
            test_gender_loss += loss_gender_fn(gender_pred, gender_label).item()

            
            correct_age += (age_pred.argmax(1) == age_label.argmax(1)).type(torch.float).sum().item()
            correct_gender += ((gender_pred >= 0.5) & (gender_label == 1) | (gender_pred < 0.5) & (gender_label == 0)).float().sum().item()

    test_age_loss /= num_batches
    test_gender_loss /= num_batches

    correct_age /= size
    correct_gender /= size

    age_val_loss_total.append(test_age_loss)
    gender_val_loss_total.append(test_gender_loss)

    age_acc_total.append(correct_age)
    gender_acc_total.append(correct_gender)
    
    # remember best acc@ and save checkpoint
    is_best = correct_age > best_acc
    if is_best:
        print('Good')
    best_acc = max(correct_age, best_acc)
    save_checkpoint({
        'age_state_dict': age_model.state_dict(),
        'gender_state_dict': gender_model.state_dict(),
    }, is_best)

    print(f"""
    Accuracy:
    age: {(100*correct_age)}%
    gender: {(100*correct_gender)}%

    Avg loss:
    age: {test_age_loss}
    gender: {test_gender_loss}
    """)