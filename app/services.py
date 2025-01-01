import time
from faster_whisper import WhisperModel
import google.generativeai as genai
from groq import Groq
from app.core.Interfaces import Service
import speech_recognition as sr 
class SpechToText(Service):
    def __init__(self, model_size:str = "medium", device:str = "cuda", compute_type:str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.rec = sr.Recognizer()
        self.listening_device = sr.Microphone()
    @staticmethod
    def callback(recognizer, audio):
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
    def start_listening(self, callback):
        with  self.listening_device as s:
            self.rec.adjust_for_ambient_noise(s, duration=2)
        print("\n Comenzando grabancion \n\n Menciona la palabra clave seguido de tu prompt")
        self.rec.listen_in_background(self.listening_device, self.callback)
        
        while True:
            time.sleep(0.5)
    def wav_to_text(self, audio_path):
        segments, _ = self.model.transcribe(audio_path)
        segments = list(segments)
        text = ''.join(seg.text for seg in segments)
        return text

    
class MainModel(Service):
    def __init__(self, convo:list[dict[str, str]], api_key):
        self.convo = convo
        self.model = Groq(api_key=api_key)
    def start(self, prompt, imgContext):
        if imgContext:
            prompt = f'USER PROMPT: {prompt}\n\n CONTEXTO DE IMGEN: {imgContext}'
        self.convo.append({"role": "user", "content": prompt})
        chat_completion = self.model.chat.completions.create(messages=self.convo, model="llama3-70b-8192")
        response = chat_completion.choices[0].message
        self.convo.append(response)
        return response.content

class ImageRecognition(Service):
    def __init__(self,api_key, generation_config:dict, safety_settings:list[dict[str, str]]):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest',
                            generation_config= generation_config,
                            safety_settings= safety_settings)
    def start(self):
        pass
    def generate_content (self, prompt, img):
        resoponse = self.model.generate_content([prompt, img])
        return resoponse.text
    