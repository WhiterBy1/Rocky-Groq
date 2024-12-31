print("Cargando librerias...")
import time
from PIL import Image
import cv2
import pyaudio
import os
import time
import speech_recognition as sr 
import re
from Prompts import Initial_prompt, function_call_prompt

print("\nLibrerias cargadas")








r = sr.Recognizer()
source = sr.Microphone()
def groq_prompt(prompt, imgContext):
    if imgContext:
        prompt = f'USER PROMPT: {prompt}\n\n CONTEXTO DE IMGEN: {imgContext}'
    convo.append({"role": "user", "content": prompt})
    chat_completion = grop_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt): 
    sys_msg = function_call_prompt()
    funcion_convo= [{'role':'system', 'content': sys_msg},
                    {'role':'user', 'content': prompt}]
    
    chat_completion = grop_client.chat.completions.create(messages=funcion_convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    
    return response.content

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt =(   

        'Eres la IA de análisis de visión que proporciona significado semántico a partir de imágenes para proporcionar contexto '
        'para enviar a otra IA que creará una respuesta para el usuario. No respondas como asistente de IA'
        'al usuario. En lugar de eso, toma el mensaje del usuario e intenta extraer todo el significado de la foto'
        'relevante para el usuario. A continuación, genera la mayor cantidad de datos objetivos sobre la imagen para la IA '
        f'asistente que responderá al usuario. \nUSER PROMPT: {prompt}'

    )
    response = model.generate_content([prompt, img])
    return response.text
# def wav_to_text(audio_path):
#     segments, _ = Whisper_model.transcribe(audio_path)
#     text = ''.join(segments.text for seg in segments)
#     return text
def wav_to_text(audio_path):
    segments, _ = Whisper_model.transcribe(audio_path)
    # Convertir el generador en una lista
    segments = list(segments)
    text = ''.join(seg.text for seg in segments)
    return text

#callback here 
def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, key_word)

    print('clean prompt: ' + prompt_text)
    new_prompt = prompt_text.lower()
    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)

        visual_context = None 

        if "take screenshot" in call:
            print("Tomando captura de pantalla")
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print("Capturando la webcam")
            webcam_path = web_cam_capture()
            if webcam_path:
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=webcam_path)
        elif 'extract clipboard' in call:
            print("Extrayendo contenido del clipboard")
            clipboard_content = get_clipboard()
            clean_prompt = f'{clean_prompt}\n\n Contenido del clipboard: {clipboard_content}'

        response = groq_prompt(prompt=clean_prompt, imgContext=visual_context)
        print(f'Assistant: {response}')

def start_listening():
    with    source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("\n Comenzando grabancion \n\n Menciona la palabra clave seguido de tu prompt")
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(0.5)

def extract_prompt(transcripted_txt, key_word):
    pattern = rf'\b{re.escape(key_word)}[\s,,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcripted_txt, re.IGNORECASE)
    
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None
    

if __name__ == '__main__':
    start_listening()
