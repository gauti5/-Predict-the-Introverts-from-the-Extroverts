from flask import Flask, render_template, request

from src.Pipelines.prediction_pipeline import predict_pipeline, CustomData

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            id=int(request.form.get('id')),
            Time_spent_Alone=float(request.form.get('Time_spent_Alone')),
            Stage_fear=object(request.form.get('Stage_fear')),
            Social_event_attendance=float(request.form.get('Social_event_attendance')),
            Going_outside=float(request.form.get('Going_outside')),
            Drained_after_socializing=object(request.form.get('Drained_after_socializing')),
            Friends_circle_size=float(request.form.get('Friends_circle_size')),
            Post_frequency=float(request.form.get('Post_frequency'))
        )
        
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        
        predictpipeline=predict_pipeline()
        result=predictpipeline.predict(pred_df)
        return render_template('result.html', final_result=result[0])
    
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)