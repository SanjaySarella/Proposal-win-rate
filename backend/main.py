from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator, ConfigDict
from backend.research import metadata, research

app = FastAPI(title='Federal Award Research', version='3.0.0')
class ResearchInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    agency: str | None = None
    award_type: str | None = None
    minimum: float = Field(0, ge=0)
    maximum: float = Field(50000000, ge=0)
    keyword: str = Field('', max_length=200)
    @model_validator(mode='after')
    def validate_filters(self):
        if self.minimum > self.maximum:
            raise ValueError('Minimum must not exceed maximum.')
        return self
@app.get('/health')
def health():
    return {'status':'ok','version':'3.0.0'}
@app.get('/metadata')
def info():
    return metadata()
@app.post('/research')
def analyze(data:ResearchInput):
    meta = metadata()
    if data.agency and data.agency not in meta['agencies']:
        raise HTTPException(422, 'Unknown agency in this snapshot.')
    if data.award_type and data.award_type not in meta['types']:
        raise HTTPException(422, 'Unknown award type in this snapshot.')
    return research(**data.model_dump())
@app.post('/predict')
def retired_prediction():
    raise HTTPException(410, 'Bid-win prediction retired: source records do not contain losing bids. Use /research.')
static = Path(__file__).resolve().parents[1] / 'static'
if static.exists():
    app.mount('/', StaticFiles(directory=static, html=True), name='frontend')
