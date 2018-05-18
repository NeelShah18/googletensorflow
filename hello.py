import tensorflow as tf

def main():
	#tf.constatn is constant: int, string, float, boolean, list or anything
	#It means we create tf.constant object of tensorflow but we can't print using print(message)!
	message = tf.constant("Hello world! I am tensorflow")
	'''
	Tesnforflow did not run the code directly as variable called.
	-> When you create any variable using ensofrlow api,
	-> it creates object of tensorflow. Now to run it you have to start a session and than you have to run that specific
	   code to run the that session to generate result. 
	'''
	session = tf.Session()
	#To print it we have to run the session using: session.run(<variable>)
	print(session.run(message))

if __name__ == '__main__':
	main()