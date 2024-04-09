import numpy as np
import os
import random
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from torchvision import transforms
from tqdm import tqdm
from model.GAENet import GAENet

import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Training a model with given datasets')
    parser.add_argument('--base-path', type=str, default='/home/jhun/Age_gender_estimation/data/UTKFace', help='Base path for the UTKFace dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--save-path', type=str, default='./pretrained', help='Path to save the trained models and checkpoints')
    args = parser.parse_args()
    
    return args

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
    
    @staticmethod
    def age_group(age):
        if age >= 100:  # 100세 이상은 모두 마지막 그룹으로 처리
            return 19
        else:
            return age // 5

    def __getitem__(self, idx):
        image_path = os.path.join(base_path, self.files[idx])
        image = Image.open(image_path)
        
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

def save_checkpoint(state, is_best, filepath):
    """
    Saves the state of the model to a file. If this is the best model based on validation,
    it saves it to a separate file.

    Args:
    - state (dict): A dictionary containing the model's state, the optimizer's state, and other metrics.
    - is_best (bool): A flag indicating whether this is the best model based on validation.
    - filepath (str): The directory path where the checkpoint will be saved.
    """
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'best.pth.tar'))
        print('Found new best model, saving...')

def update_best_model(age_mae, gender_accuracy, age_cs_at_5):

    global best_age_mae, best_gender_accuracy, best_age_cs_at_5
    is_best = False

    if age_mae < best_age_mae:
        best_age_mae = age_mae
        is_best = True
    if gender_accuracy > best_gender_accuracy:
        best_gender_accuracy = gender_accuracy
        is_best = True
    if age_cs_at_5 > best_age_cs_at_5:
        best_age_cs_at_5 = age_cs_at_5
        is_best = True

    return is_best
    
if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path
    save_path = args.save_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    algc = False

    best_age_mae = float('inf')
    best_gender_accuracy = 0
    best_age_cs_at_5 = 0
    
    files = os.listdir(base_path)
    size = len(files)
    
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

    model = GAENet().to(device)

    # loss_age_fn = nn.CrossEntropyLoss()
    loss_age_fn = nn.L1Loss()
    loss_gender_fn = nn.BCELoss()

    # Optimizer
    #import torch.optim.lr_scheduler as lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size = 10, num_workers=2)

    age_train_loss_total, gender_train_loss_total = [], []
    age_val_loss_total, gender_val_loss_total = [], []
    age_acc_total, gender_acc_total = [], []
    best_acc = 0

    for epoch in tqdm(range(EPOCHS)):
        print('Train-----')
        age_epoch_loss, gender_epoch_loss = 0, 0
        model.train()

        for batch_data, batch_labels in tqdm(train_dataloader):
            batch_data = batch_data.to(device)
            age_label = batch_labels[0].to(device, dtype=torch.float)  # 연령
            gender_label = batch_labels[1].to(device, dtype=torch.float).squeeze()  # 성별, 차원 축소

            optimizer.zero_grad()

            age_output, gender_output = model(batch_data)

            age_output = age_output.squeeze() # [batch_size, 1] -> [batch_size]
            age_loss = loss_age_fn(age_output, age_label)

            # gender_loss = loss_gender_fn(gender_output, gender_label)
            gender_loss = loss_gender_fn(gender_output, gender_label.unsqueeze(1))

            total_loss = age_loss + gender_loss
            total_loss.backward()
            optimizer.step()

            age_epoch_loss += age_loss.item()
            gender_epoch_loss += gender_loss.item()

        age_epoch_loss /= len(train_dataloader)
        gender_epoch_loss /= len(train_dataloader)

        age_train_loss_total.append(age_epoch_loss)
        gender_train_loss_total.append(gender_epoch_loss)

        print(f"Epoch {epoch}")
        print(f'Loss age:{age_epoch_loss}')
        print(f'Loss gender:{gender_epoch_loss}')

        # Vaidating
        size = len(val_dataloader.dataset)
        num_batches = len(val_dataloader)

        model.eval()
        age_cs_at_5 = 0 
        age_mae_total = 0  
        test_gender_loss = 0
        correct_gender = 0
        total_samples = 0

        for X, y in tqdm(val_dataloader, desc="Validation"):
            X = X.to(device)
            age_label = y[0].to(device, dtype=torch.float)
            gender_label = y[1].to(device, dtype=torch.float)

            age_pred, gender_pred = model(X)
            
            age_pred = age_pred.squeeze()

            age_difference = torch.abs(age_pred - age_label)
            age_cs_at_5 += torch.sum(age_difference <= 5).item()

            age_mae_total += age_difference.sum().item()

            test_gender_loss += loss_gender_fn(gender_pred, gender_label).item()

            predicted_gender = gender_pred.round()
            correct_gender += (predicted_gender == gender_label).type(torch.float).sum().item()
            total_samples += gender_label.size(0)

        age_cs_at_5_accuracy = 100 * age_cs_at_5 / total_samples
        age_mae = age_mae_total / total_samples
        test_gender_loss_avg = test_gender_loss / len(val_dataloader)
        gender_accuracy = 100 * correct_gender / total_samples

        print(f"Gender Loss: {test_gender_loss_avg:.4f}")
        print(f"Validation - Age CS@5: {age_cs_at_5_accuracy:.2f}%, Age MAE: {age_mae:.2f}, Gender Accuracy: {gender_accuracy:.2f}%")

        is_best = update_best_model(age_mae, gender_accuracy, age_cs_at_5_accuracy)
        if is_best:
            best_age_mae = age_mae
            best_gender_accuracy = gender_accuracy
            best_age_cs_at_5 = age_cs_at_5_accuracy
        
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_age_mae': best_age_mae,
            'best_gender_accuracy': best_gender_accuracy,
            'age_cs_at_5': best_age_cs_at_5,
        }
        
        save_checkpoint(state, is_best, save_path)