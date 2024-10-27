import os,sys
import yaml
from yolo_training.utils.main_utils import read_yaml_file
from yolo_training.logger import logging
from yolo_training.exception import CustomException
from yolo_training.entity.config_entity import ModelTrainerConfig
from yolo_training.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        task,
        weight,
        device,
        epocs,
        batch_size,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.task=task
        self.weight=weight
        self.device=device
        self.epochs=epocs
        self.batch_size=batch_size
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")



        try:
            if str(self.task)=="detection":
                temp="detect"
            else:
                temp="segment"

            if str(self.device)=="gpu":
                temp2=0
            else:
                temp2="cpu"  
                
                  


            logging.info("Unzipping data")
            os.system("unzip data.zip")
            os.system("rm data.zip")
           
            os.system(f"yolo task={self.task} mode=train model={self.weight} data=data.yaml epochs={self.epochs} device={temp2} imgsz=640 save=true")


            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp runs/{temp}/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
            os.makedirs(f"{self.model_trainer_config.model_trainer_dir}/results", exist_ok=True)
            os.system(f"cp runs/{temp}/train/* {self.model_trainer_config.model_trainer_dir}/results")
            
           
            os.system(f"rm -rf {self.weight}")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf test")
            os.system("rm -rf data.yaml")
            os.system("rm -rf runs")
            os.system("rm -rf README.dataset.txt")
            os.system("rm -rf README.roboflow.txt")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="artifacts/model_trainer/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            print(self.weight,"this is the weight of the model")
            print("Training complete")

            return model_trainer_artifact


        except Exception as e:
            raise CustomException(e, sys)