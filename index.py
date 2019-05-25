from Bot import contextual as ctl

def emotiondetector(cadena):
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

saved = input("Tienes un modelo creado?")
if saved == '1':    
    v = ctl.Bot()
    v.organize_data('intents2.json')
    v.remove_duplicates()
    v.trainer()
    v.neural_network(fitting=False)
else:
    h = ctl.Bot()
    h.organize_data('intentshappy.json')
    h.remove_duplicates()
    h.trainer()
    h.neural_network(fitting=True, modelname="modelhappy.tflearn")

    s = ctl.Bot()
    s.organize_data('intentssad.json')
    s.remove_duplicates()
    s.trainer()
    s.neural_network(fitting=True, modelname="modelsad.tflearn")

    n = ctl.Bot()
    n.organize_data('intentsneutral.json')
    n.remove_duplicates()
    n.trainer()
    n.neural_network(fitting=True, modelname="modelneutral.tflearn")

pregunta=''
while True:
    print("\n")
    pregunta = input('Pregunta: ')
    if pregunta != 'exit':
        emotion = emotiondetector(pregunta)
        if emotion == "happy":
            print("emotion:",emotion)
            h.response(pregunta,emotion)
        elif emotion == "sad":
            print("emotion:",emotion)
            s.response(pregunta,emotion)
        else:
            print("emotion:",emotion)
            n.response(pregunta,emotion)
    else:
        break
