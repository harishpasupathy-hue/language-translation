from fastapi import FastAPI, Request
from transformers import MarianMTModel, MarianTokenizer
import uvicorn

app = FastAPI()

MODEL_NAMES = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
}

loaded_models = {}

def get_model(lang):
    if lang not in loaded_models:
        model_name = MODEL_NAMES.get(lang)
        if not model_name:
            raise ValueError(f"Unsupported language: {lang}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[lang] = (tokenizer, model)
    return loaded_models[lang]

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.post("/translate/")
async def translate_text(request: Request):
    data = await request.json()
    text = data["text"]
    target_language = data["target_language"]

    tokenizer, model = get_model(target_language)
    translated_text = translate(text, tokenizer, model)

    return {"translated_text": translated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
