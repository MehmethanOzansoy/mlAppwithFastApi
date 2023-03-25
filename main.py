# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:13:43 2023

@author: mehme
"""
# %%
from joblib import dump, load
from typing import Union
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
labelsNames =list(dataset.target_names)

# %%
filename ="myFirstSavedModel.joblib"
neigha = load(filename)


templates = Jinja2Templates(directory="templates")

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/predict/")
async def make_prediction(request: Request,l1:float,w1:float,l2:float,w2:float):
    testData=np.array([l1,l2,w1,w2]).reshape(-1,4)
    probalities=neigha.predict_proba(testData)[0]
    predicted = np.argmax((probalities))
    probalbilty = probalities[predicted]
    predicted = labelsNames[predicted]
    return templates.TemplateResponse("prediction.html", 
                                      {"request": request , "probalities":probalities,
                                       "predicted":predicted,"probalbilty":probalbilty})


