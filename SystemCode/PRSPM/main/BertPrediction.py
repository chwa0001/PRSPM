from transformers import BertTokenizer
import numpy as np
import keras
from transformers import TFBertModel
import tensorflow as tf
import tensorflow_addons as tfa
import pickle


#fn takes 1 input which is free text and returns tokenized form for model inference
def prepare_bert_input(sentences, seq_len = 150, bert_name = 'bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',max_length=seq_len)
    input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),np.array(encodings["attention_mask"])]
    return input

def create_model(MAX_SEQ_LEN = 150, BERT_NAME = 'bert-base-uncased', N_CLASSES = 41):
    input_ids = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = keras.layers.GlobalAveragePooling1D()(last_hidden_states)
    output = keras.layers.Dense(N_CLASSES, activation="sigmoid")(avg)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

#fn to load model and weights
def load_model(weights):
    model = create_model()
    loss = keras.losses.BinaryCrossentropy()
    best_weights_file = weights
    model.load_weights(best_weights_file)
    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
    model.compile(loss=loss, optimizer=opt, metrics=[keras.metrics.AUC(multi_label=True, curve="ROC", num_labels= 41),
                                                     keras.metrics.BinaryAccuracy()])
    return model

def make_prediction(text):
    try:
        with open("./main/model/Bert/classes_41.txt", "rb") as fp: 
            classes_41 = pickle.load(fp)
        model = load_model('./main/model/Bert/bert_symptom_weights.h5')
        sentences = np.asarray([text])
        token = prepare_bert_input(sentences)
        prediction = model.predict(token)
        symptomlist = []
        thresh = 0.3
        for sentence, pred in zip(sentences, prediction):
            pred = pred.tolist()
            for score in pred:
                if score>thresh:
                    i = pred.index(score)
                    symptom = classes_41[i]
                    symptomlist.append(symptom)
        print(symptomlist)
        return symptomlist
    except Exception as e:
        print(e)
        return e

