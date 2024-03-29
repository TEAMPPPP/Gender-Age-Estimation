import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
from ultralytics import YOLO
import argparse
from datetime import datetime
import os
import csv

config = {
    'model_path_age': '/home/jhun/Age_gender_estimation/pretrained/age_model20_best_split.pth',
    'model_path_gender': '/home/jhun/Age_gender_estimation/pretrained/gender_model20_best_split.pth',
    'model_path_yolo': '/home/jhun/Age_gender_estimation/pretrained/yolov8m-face.pt'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Age and Gender Classification")
    parser.add_argument('--model_path_age', type=str, default=config['model_path_age'], help='Path to the age model.')
    parser.add_argument('--model_path_gender', type=str, default=config['model_path_gender'], help='Path to the gender model.')
    parser.add_argument('--model_path_yolo', type=str, default=config['model_path_yolo'], help='Path to the YOLO model.')
    parser.add_argument('--image_path', type=str, help='Path to the input image.')

    return parser.parse_args()

class AgeNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights = 'DEFAULT')
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        
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
        self.backbone = torchvision.models.resnet50(weights = 'DEFAULT')
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        
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

def classify_age_gender(age_model, gender_model, img, faces, device, csv_writer):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for face in faces:
        x1, y1, x2, y2, conf = face
        
        # face image crop!
        face_img = Image.fromarray(img[int(y1):int(y2), int(x1):int(x2)])
        face_img = transform(face_img).unsqueeze(0).to(device)

        age_pred = age_model(face_img).argmax(1).item() 
        gender_pred = gender_model(face_img).item()

        print(f"Age : {get_age(age_pred)}")
        print(f"Gender : {get_gender(gender_pred)}")
        
        age_text = get_age(age_pred)
        gender_text = get_gender(gender_pred)

        csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gender_text, age_text])

        text = f"Age: {age_text}, Gender: {gender_text}"    
            
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def get_age(d):
    if 0 <= d <= 19:
        return f"{d*5} - {d*5+4}"
    #if int(d) == 0:
    #    return f"{int(d) + 3}"
    #return f"{int(d)}"
    return "Unknown"

def get_gender(prob):
    if prob < 0.195:return "Male"
    else: return "Female"

def visualize_detection_cv2(model, img_path, device):
    original_img = cv2.imread(img_path)
    H, W, _ = original_img.shape

    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    results = model(img_tensor)

    detected_faces = []
    
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        boxes = result.boxes.data.cpu().numpy()
        names = result.names

        x_scale = W / 640
        y_scale = H / 640

        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_name = names[int(cls_id)]

            x1, y1, x2, y2 = [int(coord) for coord in [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]]

            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{cls_name} {conf:.2f}'
            cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            detected_faces.append((x1, y1, x2, y2, conf))

    return detected_faces 

def visualize_and_classify_faces(args, device):
    yolo_model = YOLO(args.model_path_yolo).to(device)
    age_model = AgeNeuralNetwork().to(device)
    age_model.load_state_dict(torch.load(args.model_path_age, map_location=device))
    gender_model = GenderNeuralNetwork().to(device)
    gender_model.load_state_dict(torch.load(args.model_path_gender, map_location=device))

    faces = visualize_detection_cv2(yolo_model, args.image_path, device)

    original_img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    date_folder = datetime.now().strftime("%Y%m%d")
    output_base_dir = os.path.join(os.getcwd(), 'outputs')
    output_dir = os.path.join(output_base_dir, date_folder)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image_name = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    output_image_path = os.path.join(output_dir, output_image_name)
    
    csv_file_path = os.path.join(output_dir, f"{date_folder}.csv")
    
    with open(csv_file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if os.stat(csv_file_path).st_size == 0:
            csv_writer.writerow(["Date", "Gender", "Age", "Floor", "Place of use"])
        
        classify_age_gender(age_model, gender_model, img_rgb, faces, device, csv_writer)
    
    cv2.imwrite(output_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def main():
    args = parse_arguments()
    
    device = "cpu"
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    # print(f"Using {device} device")

    visualize_and_classify_faces(args, device)
    
if __name__ == "__main__":
    main()