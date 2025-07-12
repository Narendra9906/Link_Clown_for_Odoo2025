#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 13:50:53 2025

@author: amit
"""

import os
from transformers import AutoProcessor, AutoModel, pipeline
import torch
from huggingface_hub import login


# Custom models directory
MODELS_DIR = "/home/amit/odoo_hackathon/odoo_models"

def download_remaining_models():
    """Download only the models that failed to download"""
    
    print("📥 Downloading remaining models for clothing exchange platform...")
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"📁 Models will be saved to: {MODELS_DIR}")
    
    try:
        # 1. Fix CLIP Model (the one that failed)
        print("Fixing CLIP model download...")
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
        
        # Download components separately to avoid the tokenizer issue
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODELS_DIR)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODELS_DIR)
        clip_model_weights = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODELS_DIR)
        
        # Create pipeline with explicit tokenizer
        clip_model = pipeline(
            "zero-shot-image-classification",
            model=clip_model_weights,
            tokenizer=clip_tokenizer,
            image_processor=clip_processor.image_processor
        )
        print("✅ CLIP model fixed and ready!")
        
    except Exception as e:
        print(f"❌ CLIP model error: {e}")
    
    try:
        # 2. Object Detection for detailed analysis
        print("Downloading object detector...")
        from transformers import AutoModelForObjectDetection, DetrImageProcessor
        
        detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir=MODELS_DIR)
        detr_model_weights = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=MODELS_DIR)
        object_model = pipeline(
            "object-detection",
            model=detr_model_weights,
            image_processor=detr_processor
        )
        print("✅ Object detector downloaded!")
        
    except Exception as e:
        print(f"❌ Object detector error: {e}")
    
    try:
        # 3. OCR for brand/label reading
        print("Downloading OCR model...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        ocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed",
            cache_dir=MODELS_DIR
        )
        ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed",
            cache_dir=MODELS_DIR
        )
        print("✅ OCR base model downloaded!")
        
    except Exception as e:
        print(f"❌ OCR base model error: {e}")
    
    try:
        # 4. Enhanced OCR model
        print("Downloading enhanced OCR model...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        trocr_large_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed", cache_dir=MODELS_DIR)
        trocr_large_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed", cache_dir=MODELS_DIR)
        
        easyocr_model = pipeline(
            "image-to-text",
            model=trocr_large_model,
            tokenizer=trocr_large_processor.tokenizer,
            image_processor=trocr_large_processor.image_processor
        )
        print("✅ Enhanced OCR model downloaded!")
        
    except Exception as e:
        print(f"❌ Enhanced OCR model error: {e}")
    
    print("✅ Download process completed!")
    print(f"📊 All models saved in {MODELS_DIR}")
    
    # Show downloaded models structure
    print("\n📁 Downloaded models structure:")
    for root, dirs, files in os.walk(MODELS_DIR):
        level = root.replace(MODELS_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")

if __name__ == "__main__":
    download_remaining_models()