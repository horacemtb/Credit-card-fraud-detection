from typing import Union
from pydantic import BaseModel

class Transaction(BaseModel):
    transaction: dict
    predictions: Union[dict, None] = None