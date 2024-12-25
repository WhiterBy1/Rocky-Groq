from groq import Groq
import time
from PIL import ImageGrab, Image
import cv2
import pyperclip
import google.generativeai as genai
import pyaudio
from faster_whisper import WhisperModel
import os
import speech_recognition as sr
import re

key_word = 'Asistente'
grop_client = Groq(api_key="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")
genai.configure(api_key='AIzaSyB8ptgjyzTEg-QmFqMYERQbJCSqSfwUh-0')

sys_msg = (
    'Eres asistente de voz AI multimodal. Tu usuario puede o no haber adjuntado una foto para el contexto '
    '(Una captura de pantalla o de una webcam). Cualquier foto ya ha sido procesada en un detallado '
    'prompt de texto que se adjuntará a su mensaje de voz transcrito. Genera la respuesta más útil y '
    'objetiva posible, teniendo en cuenta cuidadosamente todo el texto generado anteriormente en su respuesta antes de '
    'añadir nuevos tokens a la respuesta. No esperes ni solicites imágenes. Utiliza el contexto si se añade. '
    'Utiliza todo el contexto de esta conversación para que tu respuesta sea relevante. Haz que '
    'tus respuestas sean claras y concisas, evitando cualquier "verbosity".'
)

model_size = "medium"

# Medir el tiempo de inicio
start_time = time.time()
Whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
# Medir el tiempo de finalización
end_time = time.time()

# Calcular el tiempo transcurrido
elapsed_time = end_time - start_time
print(f"La carga de tiempo de whisper model tardó {elapsed_time} segundos en ejecutarse.")

convo = [{'role': 'system', 'content': sys_msg}]
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}
safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

r = sr.Recognizer()
source = sr.Microphone()

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Tiempo de ejecución de {func.__name__}: {end_time - start_time:.4f} segundos")
        return result
    return wrapper

@measure_time
def groq_prompt(prompt, imgContext):
    if imgContext:
        prompt = f'USER PROMPT: {prompt}\n\n CONTEXTO DE IMGEN: {imgContext}'
    convo.append({"role": "user", "content": prompt})
    chat_completion = grop_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

@measure_time
def function_call(prompt): 
    sys_msg = (
        'Eres un modelo de llamada a una función de IA. Determinarás si extraer el contenido del portapapeles del usuario, '
        'tomar una captura de pantalla, capturar la webcam o no llamar a ninguna función es lo mejor para que un asistente de voz responda '
        'al usuario. Se puede suponer que la cámara web es la de un portátil normal que mira al usuario. Se ' 
        'responder con una sola selección de esta lista: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'No respondas con nada que no sea la selección más lógica de esa lista sin explicaciones. Formatea el '
        'nombre de la llamada a la función exactamente como en lista.'
    )
    funcion_convo= [{'role':'system', 'content': sys_msg},
                    {'role':'user', 'content': prompt}]
    
    chat_completion = grop_client.chat.completions.create(messages=funcion_convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    
    return response.content

@measure_time
def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return path

@measure_time
def web_cam_capture():
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Cannot open webcam")
        return None
    
    path = "webcam.jpg"
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to capture image from webcam")
        cam.release()
        return None
    
    cv2.imwrite(path, frame)
    cam.release()
    return path

@measure_time
def get_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No hay clipboard copiado")
        return None

@measure_time
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

@measure_time
def wav_to_text(audio_path):
    segments, _ = Whisper_model.transcribe(audio_path)
    segments = list(segments)
    text = ''.join(seg.text for seg in segments)
    return text

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, key_word)

    print('clean prompt: ' + prompt_text)
    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)

        visual_context = None 

        if "take screenshot" in call:
            print("Tomando captura de pantalla")
            screenshot_path = take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path=screenshot_path)
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
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("\n Comenzando grabación \n\n Menciona la palabra clave seguido de tu prompt")
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(0.5)

def extract_prompt(transcripted_txt, key_word):
    pattern = rf'\b{re.escape(key_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcripted_txt, re.IGNORECASE)
    
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

print("\n================================")
start_listening()
