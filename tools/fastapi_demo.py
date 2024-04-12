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
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.append(base_dir)

from model.GAENet import GAENet

import warnings
warnings.filterwarnings("ignore")
                                                             
def parse_arguments():
    parser = argparse.ArgumentParser(description="Age and Gender Classification")
    parser.add_argument('--model_path_gaenet', type=str, default='/data/best.pth.tar', help='Path to GAENet model.')
    parser.add_argument('--model_path_yolo', type=str, default='/data/yolov8m-face.pt', help='Path to the YOLO model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_directory', type=str, default='./output', help='Directory to save outputs.')
    return parser.parse_args()

def classify_age_gender(gaenet_model, img, faces, device, csv_writer):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for face in faces:
        x1, y1, x2, y2, conf = face
        
        # 얼굴 이미지 크롭 및 변환
        face_img = Image.fromarray(img[int(y1):int(y2), int(x1):int(x2)])
        face_img = transform(face_img).unsqueeze(0).to(device)

        age_pred, gender_pred = gaenet_model(face_img)
        age_pred = age_pred.item()
        gender_pred = gender_pred.item() 

        print(f"Age : {get_age(age_pred)}")
        print(f"Gender : {get_gender(gender_pred)}")
        
        age_text = get_age(age_pred)
        gender_text = get_gender(gender_pred)

        csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gender_text, age_text])

        text = f"{age_text}, {gender_text}"    
            
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 6)
        cv2.putText(img, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 6)


def get_age(predicted_age):
    # predicted_age는 연령대의 인덱스를 나타냄 (0부터 시작)
    # 0이면 0-4세, 1이면 5-9세, ..., 19면 95-99세를 나타냄
    # 100세 이상은 모두 19번 인덱스로 처리됨
    
    # 예측된 인덱스를 실제 나이 구간으로 변환
    predicted_age = round(predicted_age)
    if 0 <= predicted_age < 20:
        age_start = predicted_age * 5
        age_end = age_start + 4
        return f"{age_start}-{age_end}"
    elif predicted_age == 19:  # 100세 이상 처리
        return "95"
    else:
        return "0-4"
    
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
   
def main(model_path_gaenet, model_path_yolo, image_path, output_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    yolo_model = YOLO(model_path_yolo).to(device)
    gaenet_model = GAENet().to(device)
    checkpoint = torch.load(model_path_gaenet, map_location=device)
    if "state_dict" in checkpoint:
        gaenet_model.load_state_dict(checkpoint["state_dict"])
    else:
        gaenet_model.load_state_dict(checkpoint)
    gaenet_model.eval()

    faces = visualize_detection_cv2(yolo_model, image_path, device)
    original_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    date_folder = datetime.now().strftime("%Y%m%d")
    output_base_dir = os.path.join(os.getcwd(), output_directory)
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
        
        classify_age_gender(gaenet_model, img_rgb, faces, device, csv_writer)
    
    cv2.imwrite(output_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    return output_image_path, csv_file_path

        
if __name__ == "__main__":
    args = parse_arguments()
    main(args.model_path_gaenet, args.model_path_yolo, args.image_path, args.output_directory)


