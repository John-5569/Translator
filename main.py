from fastapi import FastAPI
from pydantic import BaseModel

from transformers import MarianMTModel,MarianTokenizer

app=FastAPI()

model_name= "Helsinki-NLP/opus-mt-en-mul"

tokenizer=MarianTokenizer.from_pretrained(model_name)
model=MarianMTModel.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text:str

@app.get("/")
def home():
    return {"message ":"English to Telugu Translator"}

@app.post("/translate")
def translate(req:TranslationRequest):

    text=f">>tel<< {req.text}"
    inputs =tokenizer(text,return_tensor="pt",padding=True)

    translate=model.generate(**inputs)

    output=tokenizer.decode(translate[0],skip_special_token=True)

    return{
        "English":req.text,
        "Telugu":output
    }
