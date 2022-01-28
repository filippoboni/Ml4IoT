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

#define the function to normalize the data
def normalize(data):
    data[:,0] = (data[:,0]-9.10)*8.65
    data[:,1] = (data[:,1]-75.90)*16.56
    return data

class ModelRegistry:
    exposed = True

    def __init__(self):
        self.models_path = './models'
        try:
            os.mkdir(self.models_path)
        except FileExistsError:
            pass

        self.dht_device = adafruit_dht.DHT11(D4)

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

        ############
        print("starting measures....")
        ############

        #start the 6 measurements with 1s intervals
        data = np.zeros(shape = (6,2),dtype=np.float32) #[[0,0],[0,0],...]
        for i in range(6):
            data[i,0] = self.dht_device.temperature
            data[i,1] = self.dht_device.humidity
            time.sleep(1)

        #normalize data basing on the trained dataset in lab3
        data = normalize(data)

        data = np.expand_dims(data,axis = 0) #shape = (1,6,2)

        #load the model and make prediction
        cnn_path = "./models/{}".format(model_name)

        tflite_interpreter = tf.lite.Interpreter(model_path=cnn_path)
        tflite_interpreter.allocate_tensors()
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.set_tensor(input_details[0]['index'], data)

        tflite_interpreter.invoke()

        #prediction loop
        while True:
            now = datetime.now()
            timestamp = int(now.timestamp())
            body_temp = {
                'bn':'http://raspberrypi.local',
                'bt':timestamp,
                'e':[
                ]
            }

            body_hum = {
                'bn': 'http://raspberrypi.local',
                'bt': timestamp,
                'e': [
                ]
            }

            ##############
            print("measuring....")
            ##############

            new_measure_t = self.dht_device.temperature
            new_measure_h = self.dht_device.humidity

            ###################
            print("measured temp: {}, measured hum: {}".format(new_measure_t,new_measure_h))
            ###################
            predictions = tf.squeeze(tflite_interpreter.get_tensor(output_details[0]['index']),axis=0).numpy()

            ##################
            print("predicted_temp: {}, predicted_hum: {}".format(predictions[0],predictions[1]))
            ##################

            #check the thresholds
            if predictions[0] - new_measure_t > tthres:
                body_temp['e'].append({'n':'temperature_actual','u':'°C','t':0,'v':new_measure_t})
                body_temp['e'].append({'n': 'temperature_predicted', 'u': '°C', 't': 0, 'v': str(predictions[0])})

                body_json = json.dumps(body_temp)
                test.myMqttClient.myPublish("/sensor/temp", body_json) #send the alert to subscribers clients

            if predictions[1] - new_measure_h > hthres:
                body_hum['e'].append({'n': 'humidity_actual', 'u': '%', 't': 0, 'v': new_measure_h})
                body_hum['e'].append({'n': 'humidity_predicted', 'u': '%', 't': 0, 'v': str(predictions[1])})

                body_json = json.dumps(body_hum)
                test.myMqttClient.myPublish("/sensor/hum", body_json) #send the alert to subscribers clients

            #add the new measurements to the data window
            data = np.append(data,[[[new_measure_t,new_measure_h]]],axis=1)[:,1:,:] #shape=(6,2)
            data = normalize(data)

            #####################
            print(data)
            #####################
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