from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
import base64
import io
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import google.generativeai as genai
import psycopg2


conn = psycopg2.connect(
    dbname="sample_db",
    user="app",
    password="",
    host="informally-sought-honeybee.a1.pgedge.io",
    port="5432"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

admin_credentials = {
    "admin_username": "admin",
    "admin_password": "admin_password"  # You should store this securely, not hard-coded
}

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/PatientForm")
async def Patient_form(request: Request):
    return templates.TemplateResponse("PatientForm.html", {"request": request})

@app.get("/options")
async def ExistingResults(request: Request):
    return templates.TemplateResponse("options.html", {"request": request})

@app.get("/Visualization")
async def ExistingResults(request: Request):
    return templates.TemplateResponse("Visualization.html", {"request": request})

@app.get("/login")
async def ExistingResults(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def do_login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == admin_credentials["admin_username"] and password == admin_credentials["admin_password"]:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Predictions")
        data = cur.fetchall()
        cur.close()
        return templates.TemplateResponse("AllData.html", {"request": request, "data": data})
    else:
        # Credentials are incorrect, raise HTTPException with 401 Unauthorized status code
        raise HTTPException(status_code=401, detail="Incorrect username or password")


@app.get("/AllData")
async def get_all_data(request: Request):
    cur = conn.cursor()
    cur.execute("SELECT * FROM Predictions")
    data = cur.fetchall()
    cur.close()
    return templates.TemplateResponse("AllData.html", {"request": request, "data": data})

@app.get("/ExistingResults")
async def ExistingResults(request: Request):
    return templates.TemplateResponse("ExistingResults.html", {"request": request})


@app.get("/report")
async def report_fun(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})

@app.post("/existingresults")
async def do_login(
    request: Request,
    patientName: str = Form(...) ,dob: str = Form(...),email: str=Form(...)
):
    cur = conn.cursor()
    cur.execute("SELECT * FROM Predictions WHERE patient_name=%s and date_of_birth=%s and email=%s", (patientName,dob,email))
    existing_user = cur.fetchone()
    cur.close()
    return templates.TemplateResponse("ViewResults.html", {"request": request, "existing_user": existing_user})

@app.post("/upload")
async def report(request: Request, file: UploadFile = File(...),patientName: str = Form(...) ,dob: str = Form(...), gender: str=Form(...), email: str=Form(...)):
    s_img = await file.read()
    # Convert the bytes data to a NumPy array
    image = Image.open(io.BytesIO(s_img))

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    loaded_model = tf.keras.models.load_model('Lung.h5', compile=False)
    classes = {0: ('ca', 'colon adenocarcinoma'), 1: ('cb', 'colon benign'), 2: ('lac', 'lung adenocarcinoma'), 3: ('lb', 'lung benign'),
            4: ('lscc', 'lung squamous cell carcinoma')}
    predictions = loaded_model.predict(img_array)
    max_prob = np.max(predictions)
    class_ind = np.argmax(predictions)
    class_name = classes[class_ind]

    img_base64 = base64.b64encode(s_img).decode('utf-8')
    result = {
        "img": img_base64,
        "prediction": class_name
        
    }

    cur = conn.cursor()
    cur.execute("INSERT INTO Predictions (patient_name,date_of_birth,gender,email,prediction1,prediction2) VALUES (%s, %s,%s, %s,%s,%s)", (patientName,dob,gender,email,class_name[1],class_name[1]))
    conn.commit()
    cur.close()
    return templates.TemplateResponse("PatientForm.html", {"request": request,  "img": img_base64, "result":class_name, "patientName": patientName,"dob":dob, "gender":gender, "email":email})


@app.get("/chat", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})




@app.post("/get_gemini_completion")
def get_gemini_completion(
                            prompt: str = Form(...),  
                        ):
    try:
        gemini_api_key = ""
        genai.configure(api_key = gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        #return {"response": response.text}
        return PlainTextResponse(content=response.text, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
