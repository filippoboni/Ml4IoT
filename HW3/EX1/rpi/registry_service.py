#web service to run on raspberry

import base64
import os
from datetime import datetime
from DoSomething import DoSomething

import cherrypy
import json
import time
import numpy as np
import tensorflow as tf
from board import D4
import adafruit_dht

class ModelRegistry:
    exposed = True

    def __init__(self):
        self.models_path = './models'
        try:
            os.mkdir(self.models_path)
        except FileExistsError:
            pass

        dht_device = adafruit_dht.DHT11(D4)

#------------------EX1.2--------------------------
    def GET(self,*path,**query):
        #controls on the path
        if len(path) != 1 or path[0] != 'list':
            raise cherrypy.HTTPError('400',"Wrong path")

        #define the output body
        models = os.listdir(self.models_path)
        output_body = {
            'models':models
        }
        output_body = json.dumps(output_body)

        return output_body

#-----------------EX1.1---------------------------
    def PUT(self,*path,**query):
        #controls on the path
        if len(path) != 1 or path[0] != 'add':
            raise cherrypy.HTTPError(400,'Wrong path')

        #get the body of the client request
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)

        model_name = input_body['name']
        model_64 = input_body['model']

        #controls on the body
        if model_name is None:
            raise cherrypy.HTTPError(400,"Wrong model name")
        if model_64 is None:
            raise cherrypy.HTTPError(400,"No model")

        #decode the model from byte64 form
        model = base64.b64decode(model_64)

        #write the model in the folder ./models
        with open(self.models_path + "/" + model_name,'wb') as f:
            f.write(model)

#---------------------EX1.3------------------------------
    def POST(self,*path,**query):
        #start the mqtt service
        test = DoSomething("alertNotifier")
        test.run()

        #controls on path and query
        if len(path) != 1:
            raise cherrypy.HTTPError(400,'Wrong path')
        if len(query) != 3:
            raise cherrypy.HTTPError(400,'Wrong query')

        model_name = query['model']
        tthres = query['tthres']
        hthres = query['hthres']

        #controls on the parameters
        if model_name is None:
            raise cherrypy.HTTPError(400,'Missing model name')
        if tthres is None:
            raise cherrypy.HTTPError(400,'Missing tthres value')
        else:
            tthres = float(tthres)
        if hthres is None:
            raise cherrypy.HTTPError(400,'Missing hthres name')
        else:
            hthres = float(hthres)

        #start the 6 measurements with 1s intervals
        data = np.zeros(shape = (6,2)) #[[0,0],[0,0],...]
        for i in range(6):
            data[i,0] = dht_device.temperature
            data[i,1] = dht_device.humidity
            time.sleep(1)

        #data = np.expand_dims(data,axis = 0) #to get shape as keras requires

        #load the model and make prediction
        cnn_path = "./models/{}".format(model_name)
        cnn = tf.keras.models.load_model(path)

        #prediction loop
        while True:
            now = datetime.now()
            timestamp = int(now.timestamp())
            body = {
                'bn':'http://raspberrypi.local',
                'bt':timestamp,
                'e':[
                ]
            }

            new_measure_t = dht_device.temperature
            new_measure_h = dht_device.humidity

            predictions = cnn.predict(data)

            #check the thresholds
            if predictions[0] - new_measure_t < tthres:
                body['e'].append({'n':'temperature_actual','u':'Cel','t':0,'v':new_measure_t})
                body['e'].append({'n': 'temperature_predicted', 'u': 'Cel', 't': 0, 'v': predictions[0]})

                body_json = json.dumps(body)
                test.myMqttClient.myPublish("/276033/th_classifier", body_json) #send the alert to subscribers clients

            if predictions[1] - new_measure_h < hthres:
                body['e'].append({'n': 'humidity_actual', 'u': 'RH', 't': 0, 'v': new_measure_h})
                body['e'].append({'n': 'humidity_predicted', 'u': 'RH', 't': 0, 'v': predictions[1]})

                body = json.dumps(body)
                test.myMqttClient.myPublish("/276033/th_classifier", body_json) #send the alert to subscribers clients

            #add the new measurements to the data window
            data = np.append(data,[[new_measure_t,new_measure_h]])[1:,:] #shape=(6,2)

            time.sleep(1)

    def DELETE(self,*path,**query):
        pass

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(ModelRegistry(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()