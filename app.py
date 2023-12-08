# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup,session,redirect,url_for
import sqlite3
import numpy as np
import pandas as pd
from disease import disease_dic
from fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9
from sklearn.preprocessing import OneHotEncoder
import razorpay
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

milkquality_Recommendation_model_path = 'models/LogisticRegression.pkl'
milkquality_Recommendation_model = pickle.load(
    open(milkquality_Recommendation_model_path, 'rb'))

milk_yield_recommendation_path= 'models/milky.pkl'
milk_yield_recommendation = pickle.load(
    open(milk_yield_recommendation_path, 'rb'))

cdam_path= 'models/cd.pkl'
cdam = pickle.load(
    open(cdam_path, 'rb'))



areaprod_path= 'models/ap.pkl'
areaprod = pickle.load(
    open(areaprod_path, 'rb'))






# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key = "super secret key"

razorpay_key_id = 'rzp_test_jvBBQF2Bxdipy4'
razorpay_key_secret = '6jepv8AqDCDoWtVeQE9ZH7tM'

# render home page


@ app.route('/',methods = ["GET","POST"])
def home():
    title = 'A Farmers Touch - Home'
    msg=None
    if (request.method =="POST"):
        if(request.form["username"]!="" and request.form["password"]!=""):
            username = request.form["username"]
            password = request.form["password"]
            conn=sqlite3.connect("signup.db")
            c=conn.cursor()
            c.execute("INSERT INTO person VALUES('"+username+"','"+password+"')")
            msg="your account is created"
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        else:
            msg="something went wrong"
            
    return render_template("log.html",msg=msg , title=title)





@ app.route('/sub',methods=['POST'])
def index():
    return render_template('pindex.html')



@ app.route('/payment', methods=['POST'])
def payment():
    # Create a new instance of Razorpay client
    client = razorpay.Client(auth=(razorpay_key_id, razorpay_key_secret))

    # Get the amount from the form
    amount = int(request.form['amount']) * 100  # Convert to paise (1 INR = 100 paise)

    # Create a new order
    order = client.order.create({
        'amount': amount,
        'currency': 'INR',
        'payment_capture': '1'  # Auto capture the payment
    })

    # Fetch the order ID
    order_id = order['id']

    return render_template('payment.html', amount=amount, order_id=order_id, key_id=razorpay_key_id)
    

@ app.route("/login" , methods = ["GET","POST"])
def login():
    r=""
    m=""
    if (request.method =="POST"):
        username = request.form["username"]
        password = request.form["password"]
        conn=sqlite3.connect("signup.db")
        c=conn.cursor()
        c.execute("SELECT * FROM person WHERE username='"+username+"' and password ='"+password+"'")
        r=c.fetchall()
        for i in r:
            if(username==i[0] and password==i[1]):
                session["logedin"]=True
                session["username"]=username
                return redirect(url_for("about"))
            else:
                m="please enter valid username and password"
    return render_template("log.html", m=m)


@ app.route("/about")
def about():
    return render_template("index.html")
@ app.route("/newdash")
def newdash():
    return render_template("admindashbord.html")
    

@ app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
    
@ app.route("/feed" , methods = ["GET","POST"])
def feed():
    title = 'A Farmers Touch - feedback'
    return render_template('feedback.html', title=title)

@ app.route("/crop-production" , methods = ["GET","POST"])        
def crop_prod():
    title = 'A Farmers Touch - feedback'
    return render_template('productioncp.html', title=title)
    


# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'A Farmers Touch - Crop Recommendation'
    return render_template('crop.html', title=title)


@ app.route('/admindash', methods = ["GET","POST"])
def admindas():
    m=""
    if (request.method =="POST"):
        username = request.form["username"]
        password = request.form["password"]
        if(username=="admin" and password=="admin123"):
            return redirect(url_for("newdash"))
        else:
            m="please enter valid username and password"
    return render_template("admin.html", m=m)

            

@ app.route('/adminlogin')
def adminlog():
    return render_template('admin.html')



@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'A Farmers Touch - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@ app.route('/milk_quality_prediction')
def milk_quality_prediction():
    title = 'A Farmers Touch - Milk Quality'
    
    return render_template('milk_quality.html' , title=title)


@ app.route('/milk_yield_prediction')
def milk_yield_prediction():
    title ='A Farmers Touch - Milk Production'
    return render_template('myield.html' , title=title)

@ app.route('/crop_damage')
def crop_damage():
    title ='A Farmers Touch - Crop Damage'
    return render_template('damage.html' , title=title)
    

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page

@ app.route('/crop-prod', methods=['POST'])
def aprod():
    title = 'A Farmers Touch- Crop production'
    if request.method =='POST':
        State_Name= str(request.form['State_Name'])
        District_Name= str(request.form['District_Name'])
        Crop_Year = int(request.form['Crop_Year'])
        Season = str(request.form['Season'])
        Crop = str(request.form['Crop'])
        Area = float(request.form['Area'])
        data = np.array([[State_Name,District_Name,Crop_Year,Season,Crop,Area]])
        
        my_prediction = areaprod.predict(data)
        final_prediction = my_prediction[0]
        return render_template('area_result.html', prediction=final_prediction, title=title)
    else:
        return render_template('try_again.html', title=title)
        





@ app.route('/cropdam', methods=['POST'])
def sam():
    title = 'A Farmers Touch- Crop Damaged'
    if request.method =='POST':
        Estimated_Insects_Count = int(request.form['Estimated_Insects_Count'])
        Crop_Type = str(request.form['Crop_Type'])
        Soil_Type = str(request.form['Soil_Type'])
        Pesticide_Use_Category = str(request.form['Pesticide_Use_Category'])
        Number_Doses_Week = int(request.form['Number_Doses_Week'])
        Number_Weeks_Used = int(request.form['Number_Weeks_Used'])
        
        Season = str(request.form['Season'])
        data = np.array([[Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Season]])
        my_prediction = cdam.predict(data)
        final_prediction = my_prediction[0]
        return render_template('damage_result.html', prediction=final_prediction, title=title)
    else:
        return render_template('try_again.html', title=title)
        
        
    
@ app.route('/result-feedback', methods=['POST'])
def fbk():
    return render_template('resultfeed.html')
    

    

@ app.route('/milk-production', methods=['POST'])
def predictmilk():
    title = 'A Farmers Touch- milk quality'
    if request.method =='POST':
        parity = int(request.form['parity'])
        herd = str(request.form['herd'])
        year = int(request.form['year'])
        DIM = float(request.form['DIM'])
        MPQ = int(request.form['MPQ'])
        seasson = int(request.form['seasson'])
        SLSCC = float(request.form['SLSCC'])
        
        data = np.array([[parity,herd,year,DIM,MPQ,seasson,SLSCC]])
        my_prediction =milk_yield_recommendation.predict(data)
        final_prediction = my_prediction[0]

        return render_template('myeild_result.html', prediction=final_prediction, title=title)

    else:

        return render_template('try_again.html', title=title)








@ app.route('/milk-quality', methods=['POST'])
def milk_Qua():
    title = 'A Farmers Touch- milk quality'
    if request.method =='POST':
        pH = float(request.form['pH'])
        Temprature = int(request.form['Temprature'])
        Taste = int(request.form['Taste'])
        Odor = int(request.form['Odor'])
        Fat = int(request.form['Fat'])
        Turbidity = int(request.form['Turbidity'])
        Colour = int(request.form['Colour'])
        data = np.array([[pH,Temprature,Taste,Odor,Fat,Turbidity,Colour]])
        my_prediction = milkquality_Recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('milk_qualityres.html', prediction=final_prediction, title=title)

    else:

        return render_template('try_again.html', title=title)
        
        

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'A Farmers Touch - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'A Farmers Touch - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@ app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'A Farmers Touch - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
