from fastapi import FastAPI, HTTPException
from handler import generate_output

# Initialize FastAPI app
app = FastAPI(title="FormulaVerse API", version="1.0")


@app.post("/generate")
def generate_response(data: ExpressionRequest):
    try:
        expression = parse_expression(data.expression)
        return {"input": data.expression, "parsed": generate_output(expression)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "FormulaVerse API is running 🚀"}
