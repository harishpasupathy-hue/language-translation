from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import uvicorn

app = FastAPI(title="ML Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_MAP = {
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ml": "Helsinki-NLP/opus-mt-en-ml",
}

_loaded = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenizer_model(lang: str):
    if lang not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    if lang not in _loaded:
        model_name = MODEL_MAP[lang]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(_device)
        model.eval()
        _loaded[lang] = (tokenizer, model)
    return _loaded[lang]

def run_translate(text: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str:
    if not text.strip():
        return ""
    with torch.no_grad():
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(_device)
        generated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

class TranslationRequest(BaseModel):
    text: str
    target_language: str  # 'fr' | 'de' | 'es' | 'hi' | 'zh'

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate/")
def translate_endpoint(payload: TranslationRequest):
    try:
        tokenizer, model = get_tokenizer_model(payload.target_language)
        translated = run_translate(payload.text, tokenizer, model)
        return {"translated_text": translated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException( tus_code=500, detail="Translation failed")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
