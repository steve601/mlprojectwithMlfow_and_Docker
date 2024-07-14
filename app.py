from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model_path = 'models/model.pkl'
scaler_path ='models/scaler.pkl'

model = load_object(model_path)
scaler = load_object(scaler_path)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def prediction():
    x = [i for i in request.form.values()]
    
    columns = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin','bmi', 'diabetespedigreefunction', 'age']
    x = pd.DataFrame([x],columns=columns)
    
    x = scaler.transform(x)
    
    y_pred = model.predict(x)
    msg = 'Patient has diabetes' if y_pred == 1 else 'Patient has no diabetes'
    
    return render_template('index.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)