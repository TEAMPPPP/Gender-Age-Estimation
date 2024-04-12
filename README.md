# Gender-Age Estimation

## Installation based Conda

Follow the steps below to set up the environment:

Create a new Conda environment:
   ```bash
   conda create -n GAE python==3.8
   conda activate GAE
   cd AGE_gender_estimation
   ```
Install PyTorch and related dependencies with CUDA 11.7 support:
   ```bash
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
Install other required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

## Build Dockerfile
```bash
docker build . -t gaenet_api
docker build --no-cache -t gaenet_api . # remove cache
```
## docker run
```bash
xhost +local:docker

docker run --gpus all -it --name gaenet \
-v <your/local/directory/data>:/data \
-p 8000:8000 \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
gaenet_api bash
```
## docker run example
```bash
xhost +local:docker

docker run --gpus all -it --name gaenet \
-v /home/jhun/Age_gender_estimation_datasets_pth:/data \
-p 8000:8000 \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
gaenet_api bash
```
## UTKface Datasets (Classification: age / gender)
- download link : [https://drive.google.com/drive/folders/1IJC0Z1XyOr7E0StbeYWvPn4uTHxHIXwf?usp=drive_link](https://www.kaggle.com/datasets/jangedoo/utkface-new?rvi=1)

## Weights Downloads
Create a pretrained folder and place the pths in it.
- download link : https://drive.google.com/drive/folders/1YeYffSZx0m8m02q4KtfNXnpCg4iamqry

## Lets Start (Conda)
To train, run train.py (However, please configure the settings according to the user's interface before proceeding.)
   ```bash
   python ./tools/train.py --base-path <your/datasets/dir> --save-path <dir/you/want/to/save>
   ```

To perform test demo processing, run demo.py

   ```bash
   python ./toos/demo.py --model_path_gaenet <saved/GAENet/Weights/dir> --model_path_yolo <yolo/weights/dir> ----image_path <test/image>
   ```

To perform real-time demo processing, run Observer.py

   ```bash
   mkdir input
   ```

   ```bash
   python Observer.py
   ```
## Lets Start (Docker & FastAPI)

```bash
cd app
python3 main.py
```
- example url: (http://192.168.219.193:8000/docs#/default/read_me_Read_Me__get)

## TO-DO List
- ResNet50 test training [O]
- SwinTransformer v2 training [O]
- Dockerfile generation [O]
- FastAPI [O]
