from DoSomething import DoSomething
import datetime
import time
import json

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        events = input_json["e"]

        for event in events:
            if event['n'] == 'temperature':
                alert_type = 'Temperature Alert'
            else:
                alert_type = 'Temperature Alert'


        print("Topic: ", topic, "Prediction: ", logits, "Confidence :", prob) #(05/12/2022 19:15:01) Humidity Alert: Predicted=48.8% Actual=50.1%



if __name__ == "__main__":
    test = Subscriber("thClassifier")
    test.run()
    test.myMqttClient.mySubscribe("/276033/th_classifier")

    while True:
        time.sleep(1)

    test.end()