from enum import Enum
from pydantic import BaseModel


class PDFStatus(str, Enum):
    PROCESSING = "processing"
    DONE = "done"


class PDFFile(BaseModel):
    task_id: str
    pdf_list: list[str]
    status: PDFStatus = PDFStatus.PROCESSING
