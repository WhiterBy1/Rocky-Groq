import google.generativeai as genai
from CONST import GENAI_API_KEY
from app.core.Interfaces import Service
class Services(Service):
    def __init__(self, STTModel, CoreModel, ImageModel,  config, ):
        self.STTModel = STTModel
        self.CoreModel = CoreModel
        self.ImageModel = ImageModel
        self.config = config
    def start(self, generation_config:dict, safety_settings:list[dict[str, str]]) -> genai.GenerativeModel:
        model = genai.GenerativeModel('gemini-1.5-flash-latest',
                            generation_config= generation_config,
                            safety_settings= safety_settings)
        return model
    def configure(self):
        genai.configure(api_key=GENAI_API_KEY)