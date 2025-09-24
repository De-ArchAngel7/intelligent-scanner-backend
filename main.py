from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import fitz  # PyMuPDF
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract dependencies not available")
import pandas as pd
import re
import io
from datetime import datetime
import json
import os
import numpy as np

app = FastAPI(title="Intelligent Scanner API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    """Enhance image quality for better OCR results"""
    if not TESSERACT_AVAILABLE:
        return image  # Return original image if Tesseract not available
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Increase DPI (scale up for better OCR)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        return image
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF by converting pages to images"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""
        
        # First try to extract text directly from PDF (no OCR needed)
        for page_num in range(min(3, len(pdf_document))):  # Limit to first 3 pages
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                all_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
        
        pdf_document.close()
        
        # If we got text directly, return it
        if all_text.strip():
            return all_text.strip()
        
        # If no text found and Tesseract is available, try OCR on images
        if TESSERACT_AVAILABLE:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(min(3, len(pdf_document))):  # Limit to first 3 pages
                page = pdf_document[page_num]
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Process with OCR
                image = Image.open(io.BytesIO(img_data))
                processed_image = preprocess_image(image)
                text = pytesseract.image_to_string(processed_image, config='--psm 6')
                all_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            pdf_document.close()
            return all_text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return None

def extract_text_with_enhanced_ocr(image_bytes, file_type):
    """Enhanced OCR with preprocessing and PDF support"""
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available, using fallback extraction")
        return extract_text_fallback(image_bytes, file_type)
    
    try:
        if file_type == "application/pdf":
            result = extract_text_from_pdf(image_bytes)
            if result and len(result.strip()) > 3:
                return result
        else:
            # Handle images
            image = Image.open(io.BytesIO(image_bytes))
            processed_image = preprocess_image(image)
            text = pytesseract.image_to_string(processed_image, config='--psm 6')
            if text and len(text.strip()) > 3:
                return text.strip()
        return None
    except Exception as e:
        print(f"Enhanced OCR failed: {e}")
        return extract_text_fallback(image_bytes, file_type)

def extract_text_fallback(image_bytes, file_type):
    """Fallback text extraction when Tesseract is not available"""
    try:
        if file_type == "application/pdf":
            # Try to extract text from PDF using PyMuPDF
            doc = fitz.open(stream=image_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip() if text.strip() else None
        else:
            # For images, return a placeholder that indicates OCR is not available
            return "OCR_NOT_AVAILABLE"
    except Exception as e:
        print(f"Fallback extraction failed: {e}")
        return None

class ExtractionResult(BaseModel):
    raw_text: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    category: Optional[str] = None
    confidence: float = 0.0

class ErrorResponse(BaseModel):
    error: str
    detail: str

def extract_vendor_name(text: str) -> Optional[str]:
    """Extract vendor name with improved logic"""
    lines = text.split('\n')
    
    # Look for common business indicators
    business_indicators = ['restaurant', 'cafe', 'store', 'shop', 'company', 'inc', 'llc', 'corp']
    
    for line in lines[:8]:  # Check first 8 lines
        line = line.strip()
        if len(line) > 3 and not re.match(r'^\d+$', line):
            # Check if line contains business indicators
            if any(indicator in line.lower() for indicator in business_indicators):
                return line
            # Check if line looks like a business name (no numbers, reasonable length)
            if not re.search(r'\d{4,}', line) and 3 <= len(line) <= 50:
                return line
    
    return None

def extract_date(text: str) -> Optional[str]:
    """Extract date using various patterns"""
    date_patterns = [
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b',  # YYYY/MM/DD
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b',  # DD Mon YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})\b',  # Mon DD, YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0] if isinstance(matches[0], str) else '/'.join(matches[0])
    return None

def extract_total_amount(text: str) -> Optional[float]:
    """Extract total amount using various patterns"""
    # Look for currency patterns
    currency_patterns = [
        r'total[:\s]*\$?(\d+\.?\d*)',
        r'amount[:\s]*\$?(\d+\.?\d*)',
        r'sum[:\s]*\$?(\d+\.?\d*)',
        r'\$(\d+\.?\d*)',  # Simple dollar amount
        r'(\d+\.?\d*)\s*dollars?',
    ]
    
    for pattern in currency_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                continue
    
    # Look for the largest number that could be a total
    numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
    if numbers:
        try:
            amounts = [float(num) for num in numbers if float(num) > 0]
            if amounts:
                return max(amounts)  # Return the largest amount found
        except ValueError:
            pass
    
    return None

def categorize_invoice(text: str) -> str:
    """Simple categorization based on keywords"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['restaurant', 'food', 'dining', 'cafe', 'bar']):
        return 'Food & Dining'
    elif any(word in text_lower for word in ['gas', 'fuel', 'petrol', 'station']):
        return 'Transportation'
    elif any(word in text_lower for word in ['hotel', 'lodging', 'accommodation']):
        return 'Travel'
    elif any(word in text_lower for word in ['grocery', 'supermarket', 'store', 'market']):
        return 'Groceries'
    elif any(word in text_lower for word in ['office', 'supplies', 'stationery']):
        return 'Office Supplies'
    else:
        return 'Other'

@app.get("/")
async def root():
    return {"message": "Intelligent Scanner API is running!"}

@app.post("/extract", response_model=ExtractionResult)
async def extract_invoice_data(file: UploadFile = File(...)):
    """
    Extract structured data from uploaded invoice/receipt image or PDF
    """
    try:
        # Validate file type
        if not file.content_type.startswith(('image/', 'application/pdf')):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (PNG, JPG, JPEG) or PDF"
            )
        
        # Read file content
        content = await file.read()
        
        # Run enhanced OCR with preprocessing and PDF support
        raw_text = extract_text_with_enhanced_ocr(content, file.content_type)
        
        # Handle fallback case when Tesseract is not available
        if raw_text == "OCR_NOT_AVAILABLE":
            raise HTTPException(
                status_code=400,
                detail="Image OCR is not available on this server. Please upload a PDF file instead, which can extract text directly."
            )
        
        # If enhanced OCR fails, try basic Tesseract as fallback (only if available)
        if not raw_text or len(raw_text.strip()) < 5:
            if TESSERACT_AVAILABLE:
                print("Enhanced OCR failed, trying basic Tesseract...")
                try:
                    image = Image.open(io.BytesIO(content))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    raw_text = pytesseract.image_to_string(image, config='--psm 6')
                except Exception as e:
                    print(f"Basic OCR also failed: {e}")
                    raw_text = None
            else:
                print("Tesseract not available, cannot process images")
                raw_text = None
        
        if not raw_text or len(raw_text.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the uploaded file. Please try a clearer image or PDF."
            )
        
        # Calculate confidence based on text length and content quality
        confidence = min(0.95, 0.3 + (len(raw_text) / 1000) * 0.4)
        confidence_scores = [confidence]
        
        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded file"
            )
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Parse structured data
        vendor = extract_vendor_name(raw_text)
        date = extract_date(raw_text)
        total_amount = extract_total_amount(raw_text)
        category = categorize_invoice(raw_text)
        
        return ExtractionResult(
            raw_text=raw_text.strip(),
            vendor=vendor,
            date=date,
            total_amount=total_amount,
            category=category,
            confidence=avg_confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
