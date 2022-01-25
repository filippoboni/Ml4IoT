from DoSomething import DoSomething
import datetime
import time
import json

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        timestamp = input_json['bt']
        date = datetime.datetime.fromtimestamp(timestamp)
        events = input_json["e"]

        for event in events:
            if 'temperature' in event['n']:
                alert_type = 'Temperature Alert'
                if event['n'] == 'temperature_actual':
                    actual = event['v']
                else:
                    predicted = event['v']
            else:
                alert_type = 'Humidity Alert'
                if event['n'] == 'humidity_actual':
                    actual = event['v']*100
                else:
                    predicted = event['v']*100

            print("({:02}/{:02}/{:04} {:02}:{:02}:{:02}) {}: Predicted={}{} Actual={}{}".
                  format(date.day, date.month, date.year, date.hour,
                         date.minute, date.second,alert_type,predicted,event['u'],actual,event['u']))



if __name__ == "__main__":
    test = Subscriber("thClassifier")
    test.run()
    test.myMqttClient.mySubscribe("/276033/th_classifier")

    while True:
        time.sleep(1)

    test.end()