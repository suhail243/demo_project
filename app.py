from flask import Flask, request, render_template
import pickle
import numpy as np
import mysql.connector



# Connect to the MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",  # replace with your MySQL username
    password="suhail243",  # replace with your MySQL password
    database="demo"  # replace with your MySQL database name
)

app = Flask(__name__)

# Load the heart disease model
with open('Heart.pkl', 'rb') as file:
    heart_model = pickle.load(file)

# Load the lung disease model
with open('lungs.pkl', 'rb') as file:
    lung_model = pickle.load(file)

with open('dian.pkl', 'rb') as file:
    diabities_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('main.html')

@app.route("/d1")
def home2():
    return render_template('heart.html')

@app.route("/d3")
def home3():
    return render_template('diabites.html')

@app.route("/d2")
def home4():
    return render_template('lung.html')


@app.route("/h1")
def home5():
    return render_template('h1.html')

@app.route("/h2")
def home6():
    return render_template('h2.html')

@app.route("/h3")
def home7():
    return render_template('h3.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    # Get data from form and separate each request into a variable
    age = float(request.form['age'])
    thalach = float(request.form['thalach'])
    st_depression = float(request.form['st_depression'])
    trestbps = float(request.form['trestbps'])
    gender = float(request.form['gender'])
    chest_pain = float(request.form['chest_pain'])
    slope = float(request.form['slope'])
    thal = float(request.form['thal'])
    num_major_vessels = float(request.form['num_major_vessels'])
    exercise_induced_angina = float(request.form['exercise_induced_angina'])
    restecg = float(request.form['restecg'])

    # Combine features into a final array for prediction
    final_features = np.array([[age, thalach, st_depression, trestbps, gender, chest_pain, slope, thal, num_major_vessels, exercise_induced_angina, restecg]])
    prediction = heart_model.predict(final_features)
    
    # Ensure prediction is formatted as an integer
    prediction_result = int(prediction[0])
    
    
  
    
    return render_template('predict.html', prediction_text=prediction_result)


@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    # Get data from form and separate each request into a variable
    age = float(request.form['age'])
    thalach = float(request.form['thalach'])
    st_depression = float(request.form['st_depression'])
    trestbps = float(request.form['trestbps'])
    gender = float(request.form['gender'])
    chest_pain = float(request.form['chest_pain'])
    bns1 = float(request.form['bns1'])
    thal = float(request.form['thal'])
    num_major_vessels = float(request.form['num_major_vessels'])
    exercise_induced_angina = float(request.form['exercise_induced_angina'])
    restecg = float(request.form['restecg'])
    bns = float(request.form['bns'])

    # Combine features into a final array for prediction
    final_features = np.array([[age, thalach, st_depression, trestbps, gender, chest_pain, bns1, thal, num_major_vessels, exercise_induced_angina, restecg, bns]])
    prediction = lung_model.predict(final_features)
    
    # Ensure prediction is formatted as an integer
    prediction_result = int(prediction[0])
    
    return render_template('predict.html', prediction_text=prediction_result)




@app.route('/predict_diabit', methods=['POST'])
def predict_diabit():
    # Get data from form and separate each request into a variable
    name =float(request.form['name'])
    age = float(request.form['age'])
    gender= float(request.form['gender'])
    heart_disease = float(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    HbA1c_level= float(request.form['HbA1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    hypertension = float(request.form['hypertension'])
 
    
    BMI = bmi / (name / 100) ** 2

    
 
    
    
    

    # Combine features into a final array for prediction
    final_features = np.array([[age, heart_disease, HbA1c_level, gender, blood_glucose_level,hypertension,BMI]])
    prediction = diabities_model.predict(final_features)
    
    # Ensure prediction is formatted as an integer
    prediction_result = int(prediction[0])
    if prediction_result==1:
        prediction_result="YES"
    else:
        prediction_result="NO"
    
     
    
      # Insert data into the database
    cursor = db.cursor()
    cursor.execute("INSERT INTO users (age,gender,heart_disease,bmi,HbA1c_level,blood_glucose_level,hypertension,name,prediction_result) VALUES (%s, %s,%s,%s,%s,%s,%s,%s,%s)", (age, gender,heart_disease,bmi,HbA1c_level,blood_glucose_level,hypertension,name,prediction_result))
    db.commit()

    # Fetch data to display as output
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    
    return render_template('predict.html', prediction_text=prediction_result,users=users)



if __name__ == "__main__":
    app.run(debug=True)
