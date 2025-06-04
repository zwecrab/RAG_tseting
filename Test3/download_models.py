from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

print("Downloading image captioning model...")
VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

print("Downloading sentence transformer model...")
SentenceTransformer('all-MiniLM-L6-v2')

print("Downloading generator model...")
T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
T5Tokenizer.from_pretrained("google/flan-t5-small")

print("All models downloaded successfully!")