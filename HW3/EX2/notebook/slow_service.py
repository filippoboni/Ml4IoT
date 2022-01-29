import cherrypy
import json
import base64
import numpy as np
import tensorflow as tf

# Slow pre-process options
sampling_rate = 16000
frame_len = int(sampling_rate*0.04) # window of 40 ms
frame_step = int(sampling_rate*0.02) # step of 20 ms
low_freq = 20
upper_freq = 4000
num_mel_bins = 40
num_coefficients = 10
num_spectrogram_bins = frame_len//2+1

# Mel weight matrix
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,
                                                                    num_spectrogram_bins=num_spectrogram_bins,
                                                                    sample_rate=sampling_rate,
                                                                    lower_edge_hertz=low_freq,
                                                                    upper_edge_hertz=upper_freq)

#-------------------------------------------------PRE-PROCESSING FUNCTIONS------------------------------------------

def get_spectrogram(audio):
    """
    Compute spectrogram of the input audio
    :param audio:
    :return: spectrogram
    """
    stft = tf.signal.stft(signals=audio,
                          frame_length=frame_len,
                          frame_step=frame_step,
                          fft_length=frame_len)
    spectrogram = tf.abs(stft)

    return spectrogram

def get_mfccs(spectrogram):
    """
    Compute mel coefficients of spectrogram and return only first num_coefficients
    :param spectrogram:
    :return: mel coefficients
    """

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram+1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[:num_coefficients]

    return mfccs

def padding(audio):
    """
    Padding audio function
    :param audio:
    :return: padded audio
    """

    zero_pad = tf.zeros([sampling_rate]-tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_pad], 0)
    audio.set_shape([sampling_rate])

    return audio

def audio_process(audio_binary):
    """
    Preprocess binary audio and extract mfccs features
    :param audio_binary:
    :return: mfccs
    """
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = padding(audio)
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
        mfccs = audio_process(audio_binary)
        mfccs = np.expand_dims(mfccs, axis=0).astype(np.float32)
        
        # Load the dscnn model using the interpreter
        tflite_interpreter = tf.lite.Interpreter(model_path="./kws_dscnn_True.tflite")
        tflite_interpreter.allocate_tensors()

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        # Prepare interpreter and make prediction
        tflite_interpreter.set_tensor(input_details[0]['index'], mfccs)
        tflite_interpreter.invoke()
        output_data = tflite_interpreter.get_tensor(output_details[0]['index']).numpy()

        #############
        print("predictions logits: {}".format(output_data))
        #############
        
        #prepare the body to return to the client in json format
        inference = {
            'prediction' : int(np.argmax(output_data))
        }
        
        #convert to json
        inference = json.dumps(inference)

        ############
        print("return body: {}".format(inference))
        ############

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
    cherrypy.tree.mount(SlowService(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()






