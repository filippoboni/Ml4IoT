from DoSomething import DoSomething
from datetime import datetime
import time
import json

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        events = input_json["e"]

        for event in events:
            index = int(event["t"] / 10)

            if event["n"] == "temperature":
                data[index, 0] = float(event["v"])

            else:
                data[index, 1] = float(event["v"])

        print("Topic: ", topic, "Prediction: ", logits, "Confidence :", prob) #(05/12/2022 19:15:01) Humidity Alert: Predicted=48.8% Actual=50.1%



if __name__ == "__main__":
    test = Subscriber("thClassifier")
    test.run()
    test.myMqttClient.mySubscribe("/276033/th_classifier")

    while True:
        time.sleep(1)

    test.end()