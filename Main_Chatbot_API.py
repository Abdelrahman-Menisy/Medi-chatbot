import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
nltk.download('wordnet')
nltk.download('punkt')
from googletrans import Translator
from langdetect import detect
import uvicorn


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    msg: str

lemmatizer = WordNetLemmatizer()


intents = json.loads(open("intents (1).json").read())
 
words = pickle.load(open(r"words (1).pkl", 'rb'))
classes = pickle.load(open(r"classes (1).pkl", 'rb'))
model = load_model(r"chatbot_model_1.h5")


def clean_up_sentence(sentence):
	"""
	Takes a sentence as input, tokenizes the sentence into words, lemmatizes each word, and returns a list of lemmatized words.
	"""
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]

	return sentence_words


def bag_of_words(sentence):
	"""
	Generate a bag of words representation for the given sentence.

	Parameters:
	- sentence: a string representing the input sentence

	Return:
	- np.array: an array representing the bag of words for the given sentence
	"""
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)

	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)


def predict_class(sentence):
	# sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
	"""
	Generates predictions for the class of a given sentence using a bag of words approach.

	Parameters:
	- sentence (str): The input sentence for which the class prediction is to be generated.

	Returns:
	- list of dict: A list of dictionaries containing the predicted intent class and its probability.
	"""
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]

	ERROR_THRESHOLD = 0.25

	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

	results.sort(key=lambda x: x[1], reverse=True)

	return_list = []

	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
	return return_list


def get_response(intents_list, intents_json):
	# sourcery skip: inline-immediately-returned-variable, use-next
	"""
	get_response function: 
	- Parameters: intents_list (list), intents_json (json)
	- Return type: string
	"""
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']

	result = ''

	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result



def process_text_message(txt):
	"""
	Processes a text message by predicting the class of the message, getting a response, and returning the result.

	:param txt: The text message to process.
	:type txt: str
	:return: The response to the text message.
	:rtype: str
	"""

	global res
	predict = predict_class(txt)
	res = get_response(predict, intents)
	return res


def detect_language(text):  # sourcery skip: inline-immediately-returned-variable
    """
    A function that detects the language of the input text.

    Parameters:
    text (str): The text for which the language needs to be detected.

    Returns:
    str or None: The detected language of the text, or None if an error occurs.
    """
    try:
        language = detect(text)
        return language
    except Exception as e:
        print("Error:", e)
        return None
    

def translate_text_to_english(text, detected_language):
    """
    Translate Arabic text to English using Google Translate API.
    
    Parameters:
    arabic_text (str): The Arabic text to be translated.
    
    Returns:
    str: The translated text in English.
    """
    
    translator = Translator()
    translated_text = translator.translate(text, src=detected_language, dest='en')
    return translated_text.text

def translate_to_user_language(english_text, detected_language):

    
    translator = Translator()
    translated_text = translator.translate(english_text, src='en', dest=detected_language)
    return translated_text.text

@app.post("/medi_message")
async def process_medi_message(user_message: model_input):

    msg = user_message.msg
    detected_language = detect_language(msg)
    if detected_language == 'en':
        text_response_in_user_lang = process_text_message(msg)
    
    else:
        english_message = translate_text_to_english(msg, detected_language)
        text_response = process_text_message(english_message)
        text_response_in_user_lang = translate_to_user_language(text_response, detected_language)
    
    
    result = {
        
        "user_text": msg,
        "response": text_response_in_user_lang}
    
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("Main_Chatbot_API:app", port=8000, reload=True)
    
