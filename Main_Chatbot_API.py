import random
import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import base64
import io

nltk.download('wordnet')
nltk.download('punkt')

model = tf.keras.models.load_model(
    r"chatbot_model_v5.h5", custom_objects=None, compile=True, safe_mode=True
)

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

intents = json.loads(open(r"intents.json").read())

words = pickle.load(open(r"words_v5.pkl", 'rb'))
classes = pickle.load(open(r"classes_v5.pkl", 'rb'))

# Error messages dictionary
error_messages = {
    "no_message": "No message provided. Please enter a message.",
    "speech_recognition_error": "Unable to recognize speech. Please try again.",
    "speech_recognition_service_error": "Speech recognition service error. Please check your internet connection.",
    "general_error": "Sorry, I didn't understand that, please try again in appropriate sentence or questions :).",
    "file_format_error": "File must be in WAV format."
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    if results:
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return {"text_response": "Sorry, I didn't understand that."}
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def process_text_message(txt):
    try:
        predict = predict_class(txt)
        if not predict:
            return {"text_response": "Sorry, I didn't understand that."}
        res = get_response(predict, intents)
        return res
    except Exception as e:
        return {"text_response": error_messages["general_error"]}

def process_voice_to_text_message(audio_data):
    recognizer = sr.Recognizer()
    try:
        with io.BytesIO(audio_data) as f:
            audio = sr.AudioFile(f)
            with audio as source:
                audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='en')
        return text
    except sr.UnknownValueError:
        return {"text_response": error_messages["speech_recognition_error"]}
    except sr.RequestError:
        return {"text_response": error_messages["speech_recognition_service_error"]}
    except Exception as e:
        return {"text_response": error_messages["general_error"]}

def text_to_speech(text):
    file_name = f"output_{'en'}.mp3"
    output = gTTS(text, lang='en', slow=False)
    output.save(file_name)
    return file_name

@app.post("/medi_text")
def process_medi_message(user_message: model_input):
    try:
        text_message = user_message.msg.lower()
        if not text_message:
            return {"text_response": error_messages["no_message"]}
        text_response = process_text_message(text_message)
        voice_response = text_to_speech(text_response)
    except Exception as e:
        return {"text_response": error_messages["general_error"]}
    with open(voice_response, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")
    response_data = {"user_message": text_message, "text_response": text_response, "voice_response": encoded_content}
    return response_data

@app.post("/medi_voice")
async def process_medi_message(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.wav'):
            audio_data = await file.read()
            text_message = process_voice_to_text_message(audio_data)
            if isinstance(text_message, dict):
                return JSONResponse(content=text_message, status_code=400)
            text_message = text_message.lower()
            text_response = process_text_message(text_message)
            voice_response = text_to_speech(text_response)
        else:
            return {"text_response": error_messages["file_format_error"]}
    except Exception as e:
        return {"text_response": error_messages["general_error"]}
    with open(voice_response, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")
    response_data = {"user_message": text_message, "text_response": text_response, "voice_response": encoded_content}
    return JSONResponse(content=response_data)