from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.my_project.pipelines.prediction_pipeline import CustomData, PredictionPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Get the input values from the form
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df) 
        results = round(results[0],3)
        return render_template('home.html', results=results)
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)