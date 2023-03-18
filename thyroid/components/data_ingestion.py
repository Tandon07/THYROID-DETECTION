from thyroid import utils
from thyroid.entity import config_entity
from thyroid.entity import artifact_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
import os,sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from thyroid.config import feature_cols,TARGET_COLUMN

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)


            logging.info("collected the data with no column name")


            logging.info("column 'other' is of no use so just dropping it")
            df.drop("other",axis=1,inplace=True)

            logging.info("adding column names")
            df.columns = feature_cols


            logging.info("getting splitted val of Target column")
            target = df.target
            splitted_target = target.str.split("[^a-zA-Z]+", expand=True)
            target = splitted_target[0].replace({"":'Z'})
            df[TARGET_COLUMN]=target

            #replace "?" with Nan
            df=df.replace('?', np.nan)

            # feautre rows which is not useful
            logging.info("dropping featuree that are not useful ")
            df.drop(['TBG_measured','TBG','T3_measured','TSH measured','TT4_measured','T4U_measured','FTI_measured'],axis=1,inplace=True)


            #Create dataset folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.dataset_file_path)
            os.makedirs(dataset_dir,exist_ok=True)
            logging.info("Saving df to dataset folder")
            df=df.drop_duplicates()
            #Save df to dataset folder
            df.to_csv(path_or_buf=self.data_ingestion_config.dataset_file_path,index=False,header=True)


            # logging.info("split dataset into train and test set")
            # #split dataset into train and test set
            # train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            
            # logging.info("create dataset directory folder if not available")
            # #create dataset directory folder if not available
            # dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            # os.makedirs(dataset_dir,exist_ok=True)

            # logging.info("Save df to feature store folder")


            
            # #Save df to feature store folder
            # train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            # test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            #Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(

                dataset_file_path=self.data_ingestion_config.dataset_file_path)


            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise ThyroidException(error_message=e, error_detail=sys)
