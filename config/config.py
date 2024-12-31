
from faster_whisper import WhisperModel
from groq import Groq
import google.generativeai as genai
from CONST import GENAI_API_KEY, GROQ_API_KEY
from Prompts import Initial_prompt

# Configuración inicial


print("\nCargando api keys...")
key_word = 'luis'

grop_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GENAI_API_KEY)

print("Api keys cargadas correctamente")

sys_msg =   Initial_prompt()

print("Inizializando modelo de whisper...")
model_size = "medium"
Whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
print("Modelo de whisper cargado correctamente...")

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
                            generation_config= generation_config,
                            safety_settings= safety_settings)

