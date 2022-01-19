from MyMQTT import MyMQTT


class DoSomething():
	def __init__(self, clientID):
		# create an instance of MyMQTT class
		self.clientID = clientID
		# Give as last parameter the object itself so MyQTT.py client can invoke the notify method
		self.myMqttClient = MyMQTT(self.clientID, "mqtt.eclipseprojects.io", 1883, self) 
		


	def run(self):
		# if needed, perform some other actions befor starting the mqtt communication
		print ("running %s" % (self.clientID))
		self.myMqttClient.start()

	def end(self):
		# if needed, perform some other actions befor ending the software
		print ("ending %s" % (self.clientID))
		self.myMqttClient.stop ()

	# As said in MyMQTT.py this class must have a nofigy method
	# in which we process our messages.
	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here  
		print ("received '%s' under topic '%s'" % (msg, topic))




