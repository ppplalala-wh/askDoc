from pydantic import BaseModel


class Context(BaseModel):
    documentName: str
    pageNo: int
    content: str