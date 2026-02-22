from .Base import BaseController
from services.cv_ai_service import CVAIService
import PyPDF2, os
from Models.ResponsesEnum import ResponseSignal

class CVProcessController(BaseController):

    def __init__(self, app_setting ):
        self.app_settings = app_setting

    async def get_file_extention(self, file_id : str):
        return os.path.splitext(file_id)[-1]

    async def load_file_content(self , file_id : str):
        file_ext = await self.get_file_extention(file_id =file_id)
        file_path = os.path.join("assets/",file_id)

        if file_ext == ".pdf":
            loader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in loader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return True , text
        else:
            return False, str(ResponseSignal.FILE_EXTENTION_NOT_ALLOWED.value)

    async def cv_analysis(self, file_id:str):

        flag, file_content =  await self.load_file_content(file_id)
        if not flag :
            return False, file_content
        
        ai_service = CVAIService(self.app_settings.GROQ_API_KEY)
        result = await ai_service.analyze(file_content)
        return True, result