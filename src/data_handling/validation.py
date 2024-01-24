import pandas as pd
from logs import logger
from pydantic import BaseModel, ValidationError, validator
from typing import List


class DataSchema(BaseModel):
    text: str
    generated: int
    
    @validator('generated')
    def validate_generated(cls, value):
        if value not in [0, 1]:
            logger.error('Value of "generated" should be 0 or 1.')
            raise ValueError("Value of 'generated' should be 0 or 1.")
        return value
    
    @validator('text')
    def validate_text(cls, value):
        if not isinstance(value, str):
            logger.error('Value of "text" should be a string.')
            raise ValueError("Value of 'text' should be a string.") 

def validate_schema(df: pd.DataFrame) -> List:
    """
    Validate the schema of the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - errors (list): List of validation errors. Empty list if validation passes.
    """
    errors = []

    # Convert DataFrame to a list of dictionaries
    data_records = df.to_dict(orient='records')

    # Validate each record against the Pydantic model
    for idx, record in enumerate(data_records, start=1):
        try:
            DataSchema(**record)
        except ValidationError as e:
            error_message = f"Validation error in record {idx}: {e}"
            errors.append(error_message)
            logger.error(error_message)

    return errors
