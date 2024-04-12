FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean
RUN apt-get update

# Install required packages
RUN apt-get install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0

# Remove all the apt list files since they're not needed anymore
RUN rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages with pip
RUN pip3 install fastapi uvicorn opencv-python-headless matplotlib tqdm watchdog==4.0.0 rich==13.7.1 timm==0.9.16 ultralytics==8.1.37 python-multipart dill==0.3.8

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
