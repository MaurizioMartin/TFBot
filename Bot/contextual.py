class Bot:
    def __init__(self):
        self.intents = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?']
        self.train_x = []
        self.train_y = []
        self.model = None
        self.modelhappy = None
        self.modelsad = None
        self.modelneutral = None
        self.context = {}
        self.sentiment = {}
        
    def emotiondetector(self, cadena):
        from keras.models import load_model
        from keras.preprocessing.sequence import pad_sequences
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools, pickle
        
        MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
        
        with open('bot/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        classes = ["neutral", "happy", "sad", "hate","anger"]
        #model_test = load_model('checkpoint-0.866.h5')
        model_test = load_model('bot/checkpoint-0.909.h5')
        text=[]
        text.append(cadena)
        
        sequences_test = tokenizer.texts_to_sequences(text)
        data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
        data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
        y_prob = model_test.predict(data_test)
        for n, prediction in enumerate(y_prob):
            pred = y_prob.argmax(axis=-1)[n]
            #print(text[n],"\nPrediction:",classes[pred],"\n")
            return classes[pred]

        
    def add_ignore_words(self, ignore_words):
        """ Añade más palabras para ignorar durante el proceso """
        [self.ignore_words.append(w) for w in ignore_words if w not in self.ignore_words]
        
    def organize_data(self, json_file):
        """ carga el archivo json y lo organiza en words, classes y documents
        Cada vez que se llame esta funcion, hay que llamar a remove_duplicates"""
        import nltk
        import json
        with open(json_file, "r") as json_data:
            intents = json.load(json_data)
            # loop a través de cada sentencia de los patrones dentro de cada intento
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    #tokenize cada palabra de la sentencia
                    w = nltk.word_tokenize(pattern)
                    # lo añade a la listra de words
                    self.words.extend(w)
                    # lo añade a documents
                    self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
            self.intents = intents
            
    def remove_duplicates(self):
        """ hace el stem, minimiza cada palabra y elimina los duplicados y ordena words y classes """
        from nltk.stem.lancaster import LancasterStemmer
        stemmer = LancasterStemmer()
        words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(words)))
        self.classes = sorted(list(set(self.classes)))
    def trainer(self):
        from Bot.training import train
        self.train_x, self.train_y = train(self.documents, self.words, self.classes)

    def neural_network(self, fitting, modelname, n_epoch=1000, batch_size=4):
        from Bot.model import tensorflow_model
        if modelname == "modelhappy.tflearn":
            self.modelhappy = tensorflow_model(self.train_x, self.train_y, n_epoch, batch_size, fitting, modelname)
        elif modelname == "modelsad.tflearn":
            self.modelsad = tensorflow_model(self.train_x, self.train_y, n_epoch, batch_size, fitting, modelname)
        else:
            self.modelneutral = tensorflow_model(self.train_x, self.train_y, n_epoch, batch_size, fitting, modelname)
    
    def classify(self, sentence, emotion, show_details=False):
        """ Clasificador para la sentencia que se recibe del usuario """
        import Bot.training as training
        import json
        #genera probabilidades del modelo
        improve = False
        if emotion == "happy":
            results = self.modelhappy.predict([training.bow(sentence, self.words, show_details)])[0]
        elif emotion == "sad":
            results = self.modelsad.predict([training.bow(sentence, self.words, show_details)])[0]
        else:
            results = self.modelneutral.predict([training.bow(sentence, self.words, show_details)])[0]
        #filtra las predicciones por debajo del límite
        results = [[i,r] for i,r in enumerate(results) if r > training.ERROR_THRESHOLD]
        #ordenamos por orden de probabilidad
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        returnhelp_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        #devuelve una tupla del intento con la probabilidad
        #print("list",return_list)
        if improve and len(return_list) >=2:
            if emotion == "happy":
                json_file = "intentshappy.json"
            elif emotion == "sad":
                json_file = "intentssad.json"
            else:
                json_file = "intentsneutral.json"
            print("Ayuda a mejorar")
            for j in return_list:
                print(j)
            pos = input("which one did you mean?")
            pos = int(pos)
            returnhelp_list.append(return_list[pos])
            with open(json_file, "r+") as json_data:
                intents = json.load(json_data)
            # loop a través de cada sentencia de los patrones dentro de cada intento
                for intent in intents['intents']:
                    if intent['tag'] == returnhelp_list[0][0]:
                        intent["patterns"].append("hola")
                        #json_data.write(json.dumps(intents))
            return returnhelp_list
        return return_list
    
    def response(self, sentence, emotion, userID='123', show_details=False):
        import random
        results = self.classify(sentence, emotion)
        print(results)
        if results:
            while results:
                for i in self.intents['intents']:
                    if i['tag'] == results[0][0]:
                        return random.choice(i['responses'])
                results.pop(0)
        """ Respuesta del chat """
        """
        import random
        results = self.classify(sentence)
        #emotion = self.emotiondetector(sentence)
        #print("Prediction:",emotion,"\n")
        # si hay una clasificación se busca el tag que haga match
        if results:
            # loop mientras hayan matches que procesar
            while results:
                for i in self.intents['intents']:
                    if i['tag'] == results[0][0]:
                        print(self.context)
                        self.sentiment[userID] = emotion
                        #miro si es un context_set
                        if 'context_set' in i:
                            #miro si yo ya tengo un context y si es el mismo
                            if (userID in self.context and i['context_set'] == self.context[userID]):
                                print("Estas dentro del contexto:",i['context_set'])
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))
                            elif (userID in self.context and i['context_set'] != self.context[userID]):
                                print("Estas cambiando del contexto:",self.context[userID], 'al contexto: ',i['context_set'])
                                self.context[userID] = i['context_set']
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))
                            else:
                                print("Estas entrando al contexto:",i['context_set'])
                                self.context[userID] = i['context_set']
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))
                        #si el tag es context_filter
                        elif 'context_filter' in i:
                            #si ya estamos dentro de un contexto y si el contexto es el mismo
                            if (userID in self.context and i['context_filter'] == self.context[userID]):
                                print("Estas dentro del contexto:",i['context_filter'])
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))
                            elif (userID in self.context and i['context_filter'] != self.context[userID]):
                                print("Estas fuera del contexto:",i['context_filter'])
                                if ('good' in i['context_filter'] or 'neutral' in i['context_filter']):
                                    print("Mejorando contexto de: ",self.context[userID], "a: ", i['context_filter'])
                                    self.context[userID] = i['context_filter']
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))
                            else:
                                print("No hay contexto")
                        # encuentra un tag que haga match con el primer resultado
                        elif not 'context_filter' in i:
                            if (userID in self.context):
                                return print("No hay contexto")
                            else:
                                self.context = {}
                                print('emotion:',self.sentiment)
                                return print(random.choice(i['responses']))                           
                        else:
                            self.context = {}
                            return print("se te fue el contexto")
            results.pop(0)

    
    def responseantiguo(self, sentence, userID='123', show_details=False):
        #Respuesta del chat 
        import random
        results = self.classify(sentence)
        emotion = self.emotiondetector(sentence)
        #print("Prediction:",emotion,"\n")
        # si hay una clasificación se busca el tag que haga match
        if results:
            # loop mientras hayan matches que procesar
            while results:
                for i in self.intents['intents']:
                    if i['tag'] == results[0][0]:
                        print(self.context)
                        if 'context_set' in i:
                            print ('context:', i['context_set'])
                            self.context[userID] = i['context_set']
                            #print('userid:',userID)
                            return print(random.choice(i['responses']))
                        elif (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            print("Estas dentro del contexto:",i['context_filter'])
                            return print(random.choice(i['responses']))
                        # encuentra un tag que haga match con el primer resultado
                        elif not 'context_filter' in i:
                            self.context = {}
                            return print(random.choice(i['responses']))                           
                        else:
                            self.context = {}
                            return print("se te fue el contexto")
            results.pop(0)
    """
                       
                            
            