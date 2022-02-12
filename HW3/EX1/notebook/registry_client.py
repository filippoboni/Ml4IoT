#client side to run on the notebook

import json
import requests
import base64

#-----------------------EX1.2-----------------------
paths = {
    'cnn':'./cnn.tflite',
    'mlp': './mlp.tflite'
}

for name,path in paths.items():
    with open(path,'rb') as tflite_model:
        encoded_model = base64.b64encode(tflite_model.read())
    body = {
        'model': encoded_model,
        'name': name + '.tflite'
    }
    url = 'http://raspberrypi.local:8080/add'
    r=requests.put(url,json=body)

    #check on the response
    if r.status_code != 200:
        print("400 Error")
        print(r.text)

#-------------------------EX1.1-----------------------
url = 'http://raspberrypi.local:8080/list'
r = requests.get(url)

#check the response
if r.status_code == 200:
    rbody = r.json()
    models = rbody['models']
    if len(models) == 2:
        print (models)
    else:
        print("Wrong number of models in the folder. Found {} expected 2".format(len(models)))
else:
    print('Error')
    print(r.text)

#---------------------------EX1.3-----------------------
model_name = 'cnn.tflite'
tthres = 0.1 #in °C
hthres = 0.2 #in %
url = 'http://raspberrypi.local:8080/predict/?tthres={}&hthres={}&model={}'.format(tthres,hthres,model_name)

r = requests.post(url)

#check the response from the service
if r.status_code != 200:
    print("Error")
    print(r.text)


