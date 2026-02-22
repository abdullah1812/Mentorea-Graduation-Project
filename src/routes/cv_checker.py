import aiofiles
from fastapi import APIRouter, UploadFile, Depends, status
from controllers.CVUploadController import CvController
from controllers.CVProcessController import CVProcessController
from fastapi.responses import JSONResponse
import os
from helper.config import get_settings,  Settings
from Models.ResponsesEnum import ResponseSignal
from validations.schemas.ProcesingSchema import FileProcessingSchemaRequest

cv_rout =APIRouter(prefix="/cv_checker")

@cv_rout.post("/")
async def upload_cv(cv_file: UploadFile, app_setting: Settings = Depends(get_settings)):
    is_valid, signal = CvController().validate_uploaded_file(cv_file )

    if not is_valid:
        return JSONResponse(
            status_code= status.HTTP_400_BAD_REQUEST,
            content={
                'signal':signal
            }
        )
    
    new_file_name =  str(cv_file.filename)
    file_path = os.path.join("assets/",new_file_name)
    
    try: 
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await cv_file.read(app_setting.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal":ResponseSignal.FILE_UPLOADED_FAILD.value
            }
        )
     
    return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "signal":ResponseSignal.FILE_UPLOADED_SUCCES.value,
                "file_id": new_file_name
            }
        )

@cv_rout.post("/process/")
async def processingfile(process_reques: FileProcessingSchemaRequest, app_setting: Settings = Depends(get_settings)):

    file_id = process_reques.file_id
    process_controller = CVProcessController(app_setting)
    flag , file_content_or_error_message = await process_controller.cv_analysis(file_id=file_id)
    if not flag : 
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal":file_content_or_error_message
            }
        )
    
    file_content = file_content_or_error_message
    return file_content
