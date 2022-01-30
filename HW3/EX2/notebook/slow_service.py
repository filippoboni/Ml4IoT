import cherrypy
import json
import base64
import numpy as np
import tensorflow as tf

#preprocess audio bytes to mfcc
sampling_rate = 16000
frame_length = 640
frame_step = 320
lower_frequency = 20
upper_frequency = 4000
num_mel_bins = 40
num_coefficients = 10
num_spectrogram_bins = (frame_length) // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate,
                                                                    lower_frequency, upper_frequency)

def pad(audio):
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])
    return audio

def get_spectrogram(audio):
    stft = tf.signal.stft(audio, frame_length=frame_length,
    frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    return spectrogram

def get_mfccs(spectrogram):
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    return mfccs

def process_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = pad(audio)
    spectrogram = get_spectrogram(audio)
    mfccs = get_mfccs(spectrogram)
    mfccs = tf.expand_dims(mfccs, -1)
    return mfccs


#--------------------------------------------------------SERVICE CLASS----------------------------------------------

class SlowService(object):
    exposed = True


    def GET(self, *path, **query):
        pass
        

    def POST(self, *path, **query):
        # Recieves json file with raw audio and reads body of http request
        # Process audio to extract mfccs and make inference with slow model

        # Read body and convert json string in dictionary
        request = cherrypy.request.body.read()
        body = json.loads(request)
        
        # Extract audio and compute mfccs
        audio_str = body["e"][0]["vd"]
        audio_binary = base64.b64decode(audio_str)
        mfccs = process_audio(audio_binary)
        mfccs = np.expand_dims(mfccs, axis=0).astype(np.float32)
        
        # Load the dscnn model using the interpreter
        tflite_interpreter = tf.lite.Interpreter(model_path="./kws_dscnn_True.tflite")
        tflite_interpreter.allocate_tensors()

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        # Prepare interpreter and make prediction
        tflite_interpreter.set_tensor(input_details[0]['index'], mfccs)
        tflite_interpreter.invoke()
        output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
        
        #prepare the body to return to the client in json format
        inference = {
            'prediction' : int(np.argmax(output_data))
        }
        
        #convert to json
        inference = json.dumps(inference)

        return inference
        

    def PUT(self, *path, **query):
        pass


    def DELETE(self, *path, **query):
        pass

#-----------------------------------------------------------MAIN---------------------------------------------------


if __name__=='__main__':

    conf={'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
               'tools.sessions.on': True}
          }
    cherrypy.tree.mount(SlowService(), '/slow_service', conf)
    cherrypy.config.update({'server.socket_host': '127.0.0.1'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()






