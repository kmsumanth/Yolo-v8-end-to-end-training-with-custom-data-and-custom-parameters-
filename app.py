import os
import shutil
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yolo_training.pipeline.training_pipeline import TrainPipeline
from yolo_training.utils.main_utils import decodeImage, encodeImageIntoBase64

import uvicorn

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientApp:
    def __init__(self):
        pass

clApp = ClientApp()

class ImageRequest(BaseModel):
    image: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the yolo Traing API"}

@app.post("/train")
async def train_route(request: Request):
    try:
        dataset_url = request.query_params.get("dataset_url")
        task = request.query_params.get("task")
        pretrained_model_weight=request.query_params.get("pretrained_model_weight")
        device_type=request.query_params.get("device_type")
        epochs = int(request.query_params.get("epochs"))
        batch_size = int(request.query_params.get("batch_size"))
        
        
        if not dataset_url and not epochs and not batch_size and not task and not pretrained_model_weight and not device_type:
            raise ValueError("Missing required parameters")

        obj = TrainPipeline(dataset_url,task,pretrained_model_weight,device_type,epochs,batch_size)
        #obj.update_config_file()
        obj.run_pipeline()

        return {"message": "Training Successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
@app.get("/download")
def download_model():
    # Construct the file path
    model_dir="artifacts/model_trainer/"
    model_filename="best.pt"
    file_path = os.path.join(model_dir, model_filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Return the file as a response
    return FileResponse(file_path, media_type='application/octet-stream', filename=model_filename)    

@app.get("/clear")
def clear_data():
    # Remove the "artifacts" folder entirely
    artifacts_folder = "artifacts"
    if os.path.exists(artifacts_folder):
        shutil.rmtree(artifacts_folder)
        print(f"The folder {artifacts_folder} has been removed.")
    else:
        print(f'The folder {artifacts_folder} does not exist.')

    # Clear the contents of the "log" folder without deleting the folder itself
    log_folder = "log"
    if os.path.exists(log_folder):
        for filename in os.listdir(log_folder):
            file_path = os.path.join(log_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {log_folder} does not exist.')

    print("The data is successfully removed")
    return "Data removed successfully"


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=5000)

