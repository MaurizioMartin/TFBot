# things we need for Tensorflow
import tflearn
import tensorflow as tf

def tensorflow_model(train_x, train_y, n_epoch, batch_size, fitting, modelname):
	#we are ready to build our model
	# reset underlying graph data
	tf.reset_default_graph()

	# Build neural network
	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)

	# Define model and setup tensorboard
	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
	if fitting:
		# Start training (apply gradient descent algorithm)
		model.fit(train_x, train_y, n_epoch=n_epoch, batch_size=batch_size, show_metric=True)
		#it's better to save our data
		model.save(modelname)
	else:
		model.load('./'+modelname)
	return model