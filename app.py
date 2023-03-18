# # import pymongo

# # # Provide the mongodb localhost url to connect python to mongodb.
# # client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

# # # Database Name
# # dataBase = client["neurolabDB"]

# # # Collection  Name
# # collection = dataBase['Products']

# # # Sample data
# # d = {'companyName': 'iNeuron',
# #      'product': 'Affordable AI',
# #      'courseOffered': 'Machine Learning with Deployment'}

# # # Insert above records in the collection
# # rec = collection.insert_one(d)

# # # Lets Verify all the record at once present in the record with all the fields
# # all_record = collection.find()

# # # Printing all records present in the collection
# # for idx, record in enumerate(all_record):
# #      print(f"{idx}: {record}")



# from thyroid.logger import logging
# from thyroid.exception import ThyroidException
# from thyroid.utils import get_collection_as_dataframe
# import sys,os
# from thyroid.entity import config_entity
# from thyroid.components.data_ingestion import DataIngestion
# from thyroid.components.data_validation import DataValidation
# # from sensor.components.data_validation import DataValidation
# from thyroid.components.data_transformation import DataTransformation
# from thyroid.components.model_trainer import ModelTrainer
# from thyroid.components.model_evaluation import ModelEvaluation
# from thyroid.components.model_pusher import ModelPusher


# if __name__=="__main__":
#     try:
#         training_pipeline_config = config_entity.TrainingPipelineConfig()
#         #data ingestion
#         data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
#         print(data_ingestion_config.to_dict())
#         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
#         data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


#         # DataValidation
#         data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
#         data_validation = DataValidation(data_validation_config=data_validation_config,
#                         data_ingestion_artifact=data_ingestion_artifact)

#         data_validation_artifact = data_validation.initiate_data_validation()



# #dataValidation
#         data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
#         data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
#                         data_ingestion_artifact=data_ingestion_artifact,
#                         data_validation_artifact=data_validation_artifact)
#         data_transformation_artifact = data_transformation.initiate_data_transformation()


#         #model trainer
#         model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
#         model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
#         model_trainer_artifact = model_trainer.initiate_model_trainer()


#         #model evaluation
#         model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
#         model_eval  = ModelEvaluation(model_eval_config=model_eval_config,
#         data_validation_artifact=data_validation_artifact,
#         data_transformation_artifact=data_transformation_artifact,
#         model_trainer_artifact=model_trainer_artifact)
#         model_eval_artifact = model_eval.initiate_model_evaluation()
        



#         #model pusher
#         model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
        
#         model_pusher = ModelPusher(model_pusher_config=model_pusher_config, 
#                 data_transformation_artifact=data_transformation_artifact,
#                 model_trainer_artifact=model_trainer_artifact)
#         model_pusher_artifact = model_pusher.initiate_model_pusher()

#     except Exception as e:
#         raise ThyroidException(e, sys)



from flask import Flask, request, render_template
import pickle
import pandas as pd
from thyroid.utils import load_object
from thyroid.predictor import ModelResolver

model_resolver = ModelResolver(model_registry="saved_models")
# Load the trained model, target encoder and transformer
transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
model = load_object(file_path=model_resolver.get_latest_model_path())
target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

app = Flask(__name__)

# Define a route to handle incoming requests from users
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTML form
    input_data = {
        'age': float(request.form['age']),
        'sex': request.form['sex'],
        'on_thyroxine': request.form['on_thyroxine'],
        'query_on_thyroxine': request.form['query_on_thyroxine'],
        'on_antithyroid_medication': request.form['on_antithyroid_medication'],
        'sick': request.form['sick'],
        'pregnant': request.form['pregnant'],
        'thyroid_surgery': request.form['thyroid_surgery'],
        'I131_treatment': request.form['I131_treatment'],
        'query_hypothyroid': request.form['query_hypothyroid'],
        'query_hyperthyroid': request.form['query_hyperthyroid'],
        'lithium': request.form['lithium'],
        'goitre': request.form['goitre'],
        'tumor': request.form['tumor'],
        'hypopituitary': request.form['hypopituitary'],
        'psych': request.form['psych'],
        'TSH': float(request.form['TSH']),
        'T3': float(request.form['T3']),
        'TT4': float(request.form['TT4']),
        'T4U': float(request.form['T4U']),
        'FTI': float(request.form['FTI'])
    }
    
    # Transform the input data using the target encoder and transformer
    input_df = pd.DataFrame(input_data, index=[0])
                                 # transformed_data = target_encoder.transform(input_df)
    transformed_data = transformer.transform(transformed_data)
    
    # Use the transformed data as input to the model and get the predicted output
    prediction = model.predict(transformed_data)
    cat_prediction = target_encoder.inverse_transform(prediction)



    # Assign the corresponding category based on the predicted output
    if cat_prediction in ['A', 'B', 'C', 'D']:
        category = 'hyperthyroid conditions'
    elif cat_prediction in ['E', 'F', 'G', 'H']:
        category = 'hypothyroid conditions'
    elif cat_prediction in ['I', 'J']:
        category = 'binding protein'
    elif cat_prediction == 'K':
        category = 'general health'
    elif cat_prediction in ['L', 'M', 'N']:
        category = 'replacement therapy'
    elif cat_prediction == 'R':
        category = 'discordant results'
    else:
        category = 'unknown'


# Render the predicted category on the HTML template
    return render_template('result.html', category=category)

@app.route('/')
def form():
    return render_template('form.html')

if __name__ == '__main__':
    name = 'Flask app'
    app.run(debug=True,port=5600)

