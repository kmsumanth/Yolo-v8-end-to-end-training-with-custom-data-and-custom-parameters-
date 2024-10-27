import sys, os
import yaml
from yolo_training.logger import logging
from yolo_training.exception import CustomException
from yolo_training.components.data_ingestion import DataIngestion
from yolo_training.components.data_validation import DataValidation
from yolo_training.components.model_trainer import ModelTrainer
from yolo_training.constant import training_pipeline


from yolo_training.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig,
                                                 ModelTrainerConfig)

from yolo_training.entity.artifacts_entity import (DataIngestionArtifact,
                                                    DataValidationArtifact,
                                                    ModelTrainerArtifact)


class TrainPipeline:
    def __init__(self,dataset_url:str,task:str,pretrained_model_weight:str,device_type:str,epochs:int,batch_size:int):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.dataset_url=dataset_url
        self.task=task
        self.pretrained_model_weight=pretrained_model_weight
        self.device_type=device_type
        self.epochs=epochs
        self.batch_size=batch_size

    # def update_config_file(self):
    #     file_name = "yolo_training/config.yaml"
        
    #     with open(file_name, "r") as f:
    #         config = yaml.safe_load(f)
        
    #     # Update the config dictionary with new values
    #     config['DATA_DOWNLOAD_URL'] = self.dataset_url
    #     config['TASK'] = self.task
    #     config['PRETRAINED_WEIGHT_NAME'] = self.pretrained_model_weight
    #     config['DEVICE_TYPE'] = self.device_type
    #     config['NO_EPOCHS'] = self.epochs
    #     config['BATCH_SIZE'] = self.batch_size
        
    #     with open(file_name, "w") as f:
    #         yaml.safe_dump(config, f)
            

        
    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try: 
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                self.dataset_url,
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
        


    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e



    
    def start_model_trainer(self
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                self.task,
                self.pretrained_model_weight,
                self.device_type,
                self.epochs,
                self.batch_size,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
        

        

    

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            if data_validation_artifact.validation_status == True:
                model_trainer_artifact = self.start_model_trainer()
            
            else:
                raise Exception("Your data is not in correct format")

        
        except Exception as e:
            raise CustomException(e, sys)

"""if __name__=="__main__":
    obj= TrainPipeline()
    obj.pass_values()
    obj.run_pipeline()"""