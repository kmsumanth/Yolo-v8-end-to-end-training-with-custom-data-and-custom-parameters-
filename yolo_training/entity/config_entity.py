import os
from dataclasses import dataclass
from datetime import datetime
from yolo_training.constant.training_pipeline import *



# import yaml

# # Define the path to your config file
# config_file = "yolo_training/config.yaml"

# # Load the configuration from the YAML file
# with open(config_file, "r") as f:
#     config = yaml.safe_load(f)

# # Access the configuration values
# DATA_DOWNLOAD_URL = config.get('DATA_DOWNLOAD_URL')
# TASK = config.get('TASK')
# MODEL_TRAINER_PRETRAINED_WEIGHT_NAME = config.get('PRETRAINED_WEIGHT_NAME')
# DEVICE_TYPE = config.get('DEVICE_TYPE')
# MODEL_TRAINER_NO_EPOCHS = config.get('NO_EPOCHS')
# MODEL_TRAINER_BATCH_SIZE = config.get('BATCH_SIZE')





@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR



training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig() 


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )

    # data_download_url: str = DATA_DOWNLOAD_URL



@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATUS_FILE)

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES





@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME
    )

    # weight_name = MODEL_TRAINER_PRETRAINED_WEIGHT_NAME

    # no_epochs = MODEL_TRAINER_NO_EPOCHS

    # batch_size = MODEL_TRAINER_BATCH_SIZE

    # task = TASK

    # device_type = DEVICE_TYPE