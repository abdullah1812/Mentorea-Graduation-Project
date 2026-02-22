from .Base import BaseController
from fastapi import UploadFile
from Models.ResponsesEnum import ResponseSignal

class CvController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 

    def validate_uploaded_file(self, file:UploadFile):

        if file.content_type not in self.app_settings.FILE_AVAILABLE_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        
        current_position = file.file.tell()
        file.file.seek(0, 2)  
        file_size = file.file.tell()
        file.file.seek(current_position)
        
        if file_size > self.app_settings.FILE_MAX_SIZE:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseSignal.FILE_VALIDATED_SUCCES.value
        

        