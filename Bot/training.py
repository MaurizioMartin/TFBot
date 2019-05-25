import numpy as np
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

ERROR_THRESHOLD = 0.25

def train(documents, words, classes):
	"""Unfortunately the data structure we have won’t work with Tensorflow. 
	We need to transform it further: from documents of words into tensors of numbers."""
	training = []
	output = []
	# create an empty array for our output
	output_empty = [0] * len(classes)
	# training set, bag of words for each sentence
	for doc in documents:
		# initialize our bag of words
		bag = []
		# list of tokenized words for the pattern
		pattern_words = doc[0]
		# stem each word
		pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

		# create our bag of words array
		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)
		# output is a '0' for each tag and '1' for current tag
		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1
		training.append([bag, output_row])

	# shuffle our features and turn into np.array
	random.shuffle(training)
	training = np.array(training)

	# create train and test lists
	train_x = list(training[:,0])
	train_y = list(training[:,1])
	return train_x, train_y
	"""Notice that our data is shuffled. 
	Tensorflow will take some of this and use it as test data to gauge accuracy for a newly fitted model."""

def clean_up_sentence(sentence):
	# tokenize the pattern
	sentence_words = nltk.word_tokenize(sentence)
	# stem each word
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
	# tokenize the pattern
	sentence_words = clean_up_sentence(sentence)
	# bag of words
	bag = [0]*len(words)
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print ("found in bag: %s" % w)

	return(np.array(bag))





