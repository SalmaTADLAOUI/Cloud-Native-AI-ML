import pickle

import numpy as np
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeRegressor
app = Flask(__name__)
with open('C:/Users/Salma/Desktop/S3_master/cloud_native_ai_kelloubi/seance_fastAPI/Lab folder - FastAPI Pydantic/Lab folder - FastAPI Pydantic/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    '''
    pour l'affichage sur html
    '''

    features = request.form.to_dict()
    features = list(features.values())
    features = list(map(float, features))
    print(features)
    final_features = np.array(features).reshape(1,6)
    prediction = model.predict(final_features)

    #select = request.form.get('category')
    output = prediction[0]

    return render_template('index.html', prediction_text='Breast Cancer prediction is :  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
