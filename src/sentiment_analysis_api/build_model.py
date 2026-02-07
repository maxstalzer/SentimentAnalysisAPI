from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import os

# 1. Select a Multilingual SENTIMENT Model
# This model is distilled (small) and trained on 10+ languages (including Danish/English)
# Labels: "positive", "neutral", "negative"
model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
save_directory = "tiny_model_onnx"

print(f"Downloading and Exporting {model_id} to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(
    model_id, 
    export=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Quantize (Shrink to int8)
print("Quantizing model...")
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)

quantizer.quantize(
    save_dir=save_directory,
    quantization_config=qconfig,
)
tokenizer.save_pretrained(save_directory)

# Clean up the large non-quantized file to save space
if os.path.exists(f"{save_directory}/model.onnx"):
    os.remove(f"{save_directory}/model.onnx")

print(f"Done! Model saved to {save_directory}/")