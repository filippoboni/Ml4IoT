#web service to run on raspberry

import base64
import os
import cherrypy
import json

class ModelRegistry:
    exposed = True

    def __init__(self):
        self.models_path = './models'

        try:
            os.mkdir(self.models_path)
        except FileExistsError:
            pass

#------------------EX1.1--------------------------
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

#-----------------EX1.2---------------------------
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
            raise cherrypy.HTTPError('400',"Wrong model name")
        if model_64 is None:
            raise cherrypy.HTTPError('400',"No model")

        #decode the model from byte64 form
        model = base64.b64decode(model_64)

        #write the model in the folder ./models
        with open(self.models_path + "/" + model_name,'wb') as f:
            f.write(model)

    def POST(self,*path,**query):
        pass

    def DELETE(self,*path,**query):
        pass