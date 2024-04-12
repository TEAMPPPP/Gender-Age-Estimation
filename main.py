from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
from datetime import datetime
import sys
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.append(base_dir)

from tools.fastapi_demo import main

app = FastAPI()

@app.get("/Read-Me/", response_class=HTMLResponse)
def read_me():
    """
    Perform Age-Gender Estimation as follows:

    - **Read ME**:
        - Click ‘Try it out’ next to Parameters'
        - Click 'Execute'
        - Read the explanation in the response body

    """
    # Define the path to the HTML file
    html_file_path = "/app/static/instruction.html"
    
    # Check if the file exists
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="Instruction file not found")
    
    # Read the content of the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Return the content as HTML response
    return HTMLResponse(content=html_content)


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):

    try:
        today = datetime.now().strftime("%Y%m%d")
        temp_dir = f"./temp/{today}"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model_path_gaenet = '/data/best.pth.tar'
        model_path_yolo = '/data/yolov8m-face.pt'
        output_directory = f'./output/'
        os.makedirs(output_directory, exist_ok=True)
        
        output_image_path, output_csv_path = main(model_path_gaenet, model_path_yolo, image_path, output_directory)

        return {
            "message": "Image processed successfully.",
            "image_url": output_image_path,
            "csv_url": output_csv_path
        }
    except Exception as e:
        return {"error": str(e)}

def get_latest_file(directory, pattern):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(pattern)]
    if files:
        return max(files, key=os.path.getmtime)
    return None

@app.get("/download-image/")
async def download_image():
    today = datetime.now().strftime("%Y%m%d")
    directory = f'./output/{today}'
    latest_image_file = get_latest_file(directory, ".jpg")
    if latest_image_file:
        return FileResponse(path=latest_image_file, media_type="image/jpeg", filename=os.path.basename(latest_image_file))
    raise HTTPException(status_code=404, detail="No image file found.")

@app.get("/download-csv/")
async def download_csv():
    today = datetime.now().strftime("%Y%m%d")
    directory = f'./output/{today}'
    latest_csv_file = get_latest_file(directory, ".csv")
    if latest_csv_file:
        return FileResponse(path=latest_csv_file, media_type="application/octet-stream", filename=os.path.basename(latest_csv_file))
    raise HTTPException(status_code=404, detail="No CSV file found.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
