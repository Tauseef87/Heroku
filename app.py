import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

#The below route will render index.html 
@app.route('/')
def home():
    return render_template('index.html')

#The below route will work for predict button
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #This part takes list of textbox values from UI.
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

#The below part run the whole flask
if __name__ == "__main__":
    app.run(debug=True)