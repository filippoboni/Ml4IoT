#client side to run on the notebook

import json
import requests
import base64

#-----------------------EX1.1-----------------------
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
    url = 'http://0.0.0.0:8080/add'
    r=requests.put(url,json=body)

    #check on the response
    if r.status_code != 200:
        print("400 Error")
        print(r.text)

#-------------------------EX1.2-----------------------
url = 'http://0.0.0.0:8080/list'
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

