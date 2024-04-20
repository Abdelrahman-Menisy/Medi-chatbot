import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
nltk.download('wordnet')
nltk.download('punkt')
import speech_recognition as sr
from gtts import gTTS
from fastapi import FastAPI, HTTPException, UploadFile, File
import base64
from pydub import AudioSegment



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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


def text_to_speech(text):
	# sourcery skip: inline-immediately-returned-variable
  """
  Generates a text-to-speech audio file from the input text in the specified language.

  Parameters:
    text (str): The text to be converted to speech.
    language (str): The language in which the text should be spoken.

  Returns:
    str: The filename of the saved audio file.
  """
  file_name = f"output_{'en'}.mp3"
  output = gTTS(text, lang='en', slow=False)
  output.save(file_name)
  return file_name

def check_file_type(file: UploadFile) -> str:
  """
  Checks the content type of the uploaded file.

  Args:
      file (UploadFile): The uploaded file as a FastAPI UploadFile object.

  Returns:
      str: The file type ("text" or "audio").

  Raises:
      HTTPException: If the file type is not supported.
  """

  # Check the content type of the file
  if file.content_type is not None:
    content_type = file.content_type.lower()

    # Determine the file type
    if content_type in ["text/plain", "text/csv"]:
        return "text"
    elif content_type in ["audio/mpeg", "audio/wav", "audio/ogg"]:
        return "audio"
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Only text or audio files allowed.")
  else:
    raise HTTPException(status_code=400, detail="No file uploaded.")


def process_voice_to_text_message(audio_data):
	# sourcery skip: inline-immediately-returned-variable, raise-from-previous-error
    """
    A function that processes audio data to convert it into text using Google's Speech Recognition API.

    Parameters:
    - audio_data: The audio data to be converted to text.
    - detected_language: The language detected in the audio data.

    Returns:
    - The recognized text from the audio data.
    - Raises HTTPException with status code 400 if speech recognition fails to recognize the speech.
    - Raises HTTPException with status code 500 if there's an error with the speech recognition service.
    """
    recognizer = sr.Recognizer()
    try:
        # Recognize speech using Google's Speech Recognition API
        text = recognizer.recognize_google(audio_data, language='en')
        return text
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Unable to recognize speech")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Speech recognition service error: {str(e)}")
    
def convert_to_wav(source_file, destination_file="converted.wav"):
  """
  Converts an audio file to WAV format using pydub.

  Args:
      source_file (str): Path to the source audio file.
      destination_file (str, optional): Path to the output WAV file. Defaults to "converted.wav"
  """

  # Load the audio file
  audio = AudioSegment.from_file(source_file)

  # Convert to WAV format
  audio.export(destination_file, format="wav")

  return destination_file
    

@app.post("/medi_message")
async def process_medi_message(file: UploadFile = File(...)):

    file_type = check_file_type(file)
    try:
        if file_type == "text":
            # Process text file (e.g., read and store content)
            text_data = await file.read()
            text_data = text_data.decode("utf-8")
            text_response = process_text_message(text_data)
            voice_response = text_to_speech(text_response)
            
        elif file_type == "audio":
            # Process audio file
            audio_data = await file.read()
            # Convert to WAV format
            if file.filename.endswith('.wav'):
                destination_file = file.filename
            else:
                destination_file = convert_to_wav(file.filename)
            
            text_message = process_voice_to_text_message(audio_data)
            text_response = process_text_message(text_message)
            voice_response = text_to_speech(text_response)
            
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
    # Encode the audio data
    with open(voice_response, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")
    
    
    # Combine text and audio into a single JSON response
    response_data = {"user_message": text_data, "text_response": text_response, "voice_response": encoded_content}
        
    
    return JSONResponse(content=response_data)


    
