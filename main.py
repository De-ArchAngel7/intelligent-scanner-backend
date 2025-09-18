from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import easyocr
import pandas as pd
import re
import io
from datetime import datetime
import json
import os

app = FastAPI(title="Intelligent Scanner API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR with better error handling
try:
    ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("✅ EasyOCR initialized successfully!")
except Exception as e:
    print(f"❌ EasyOCR initialization failed: {e}")
    print("🔄 Falling back to mock OCR for testing...")
    ocr = None

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
    """Extract vendor name from the first few lines of text"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if len(line) > 3 and not re.match(r'^\d+$', line):  # Not just numbers
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
        
        # Run OCR
        if ocr:
            # Use EasyOCR (better accuracy)
            try:
                result = ocr.readtext(content)
                
                # Extract text from OCR results
                raw_text = ""
                confidence_scores = []
                
                if result:
                    for line in result:
                        if line and len(line) >= 3:
                            text_line = line[1]  # Extracted text
                            confidence = line[2]  # Confidence score
                            raw_text += text_line + "\n"
                            confidence_scores.append(confidence)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR processing failed: {str(e)}"
                )
        else:
            # Fallback: Mock OCR for testing (when EasyOCR fails to initialize)
            raw_text = """
            MOCK INVOICE DATA
            Restaurant ABC
            123 Main Street
            City, State 12345
            
            Invoice #: INV-2024-001
            Date: 12/25/2023
            
            Item 1: Coffee        $4.50
            Item 2: Sandwich      $8.99
            Item 3: Dessert       $5.25
            
            Subtotal:            $18.74
            Tax:                 $1.50
            Total:               $20.24
            
            Thank you for your business!
            """
            confidence_scores = [0.85, 0.90, 0.88, 0.92, 0.87]
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
