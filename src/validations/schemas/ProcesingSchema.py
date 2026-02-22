from pydantic import BaseModel
from typing import Optional

class FileProcessingSchemaRequest(BaseModel):

    file_id : str