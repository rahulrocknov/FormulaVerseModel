from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from simple_parser import parse_expression   # your parser function

# Initialize FastAPI app
app = FastAPI(title="FormulaVerse API", version="1.0")

# Load model and tokenizer once at startup
MODEL_PATH = "./model"  # path to your fine-tuned model folder
print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded successfully on {device.upper()}")

# Input schema for requests
class ExpressionRequest(BaseModel):
    expression: str

@app.post("/generate")
def generate_response(data: ExpressionRequest):
    try:
        # 1️⃣ Parse input using your parser
        parsed_input = parse_expression(data.expression)
        if not parsed_input:
            raise HTTPException(status_code=400, detail="Invalid expression format")

        # 2️⃣ Tokenize input
        inputs = tokenizer(
            parsed_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        # 3️⃣ Generate output from model
        outputs = model.generate(**inputs, max_new_tokens=100)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4️⃣ Send response back
        return {"input": data.expression, "parsed": parsed_input, "output": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "FormulaVerse API is running 🚀"}
