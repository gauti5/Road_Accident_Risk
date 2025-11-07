from flask import Flask, render_template, request
import sys, os

from Pipelines.prediction_pipeline import predict_pipeline, CustomData
from src.exception import CustomException

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', "POST"])

       
       
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        def to_bool(val):
            if val is None:
                return False
            return str(val).strip().lower() in ['true', '1', 'yes', 'y']
        try:
            
            data=CustomData(
                road_type=request.form.get('road_type'),
                num_lanes=int(request.form.get('num_lanes')),
                curvature=float(request.form.get('curvature')),
                speed_limit=int(request.form.get('speed_limit')),
                lighting=request.form.get('lighting'),
                weather=request.form.get('weather'),
                road_signs_present=bool(request.form.get('road_signs_present')),
                public_road=bool(request.form.get('public_road')),
                time_of_day=request.form.get('time_of_day'),
                holiday=bool(request.form.get('holiday')),
                school_season=bool(request.form.get('school_season')),
                num_reported_accidents=int(request.form.get('num_reported_accidents'))
            )
            
            pred_df=data.get_data_as_a_frame()
            print(pred_df)
            predictpipeline=predict_pipeline()
            result=predictpipeline.predict(pred_df)
            return render_template("result.html", final_result=result[0])
        except Exception as e:
            raise CustomException(e,sys)
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
        