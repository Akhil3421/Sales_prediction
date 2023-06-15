import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
#load the model
model=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
encoderr=pickle.load(open('encoder.pkl','rb'))
# with open('encoding.pkl', 'rb') as f:
#     encoder = pickle.load(f)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    new_data=stdd(data)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

def stdd(data):
    column_names = ['IsHoliday', 'Type', 'Store', 'Dept', 'Date', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
    df = pd.DataFrame([data], columns=column_names)
    nums = df.drop(columns=['IsHoliday','Type'])
    nums['Date']=nums['Date'].str.replace('-','')
    nums['Date']=nums['Date'].astype(float)
    nums['Date']
    nums= nums.astype(float)
    new_cat=pd.DataFrame(encoderr.transform([[df['IsHoliday'][0], df['Type'][0]]]))
    new_cat=new_cat.astype(float)
    new_nums=scaler.transform(nums)
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums)],axis=1)
    return new_data


@app.route('/predict',methods=['POST'])
def predict():
    #data = [request.form[x] for x in ['Date', 'IsHoliday', 'Type', 'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']]    
    data=[x for x in request.form.values()]
    final_input=stdd(data)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Sales predicted price is {}".format(output))



if __name__=="__main__":
    app.run()