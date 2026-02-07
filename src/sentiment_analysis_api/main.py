from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

app = FastAPI()

# Load the local quantized model
model_path = "tiny_model_onnx"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    file_name="model_quantized.onnx"
)

class TextInput(BaseModel):
    text: str

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.post("/v1/sentiment")
def analyze_sentiment(input_data: TextInput):
    # 1. Tokenize (Keep max_length=128 for speed)
    inputs = tokenizer(
        input_data.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    )
    
    # 2. Inference
    outputs = model(**inputs)
    
    # 3. Get Probabilities
    logits = outputs.logits[0].detach().numpy()
    probs = softmax(logits)
    
    # Map probabilities to variables (CHECK YOUR MODEL'S ID2LABEL MAPPING!)
    # For lxyuan/distilbert-base-multilingual-cased-sentiments-student:
    # 0 = positive, 1 = neutral, 2 = negative
    prob_pos = float(probs[0])
    prob_neu = float(probs[1])
    prob_neg = float(probs[2])
    
    # 4. Winner-Takes-All Scoring
    # Find which bucket has the highest probability
    winner_index = np.argmax(probs)
    
    final_score = 0.0

    if winner_index == 0:  # POSITIVE WINS
        # Base score 3, plus extra confidence
        # Range: 3.0 to 5.0
        final_score = 3.0 + (prob_pos * 2.0)
        
    elif winner_index == 2: # NEGATIVE WINS
        # Base score -3, minus extra confidence
        # Range: -3.0 to -5.0
        final_score = -3.0 - (prob_neg * 2.0)
        
    else: # NEUTRAL WINS (Index 1)
        # Keep it close to 0, but allow slight drift based on bias
        # Range: -0.5 to 0.5
        final_score = (prob_pos - prob_neg)

    # 5. Final Clamp
    final_score = max(-5, min(5, final_score))

    return {"score": float(final_score)}