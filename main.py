from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
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

def extract_text_from_image_api(image_bytes):
    """Extract text from image using free OCR API"""
    try:
        # Use OCR.space free API (1000 requests per day)
        api_key = "helloworld"  # Free API key
        url = "https://api.ocr.space/parse/image"
        
        # Prepare the image
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Make API request
        response = requests.post(
            url,
            files={"file": ("image.png", img_buffer, "image/png")},
            data={
                "apikey": api_key,
                "language": "eng",
                "isOverlayRequired": False,
                "detectOrientation": True,
                "scale": True,
                "OCREngine": 2
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("IsErroredOnProcessing", False):
                print(f"OCR API error: {result.get('ErrorMessage', 'Unknown error')}")
                return None
            
            parsed_results = result.get("ParsedResults", [])
            if parsed_results:
                text = parsed_results[0].get("ParsedText", "")
                return text.strip() if text.strip() else None
        
        print(f"OCR API failed with status: {response.status_code}")
        return None
        
    except Exception as e:
        print(f"OCR API failed: {e}")
        return None

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF by converting pages to images"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""
        
        # Extract text directly from PDF (no OCR needed)
        for page_num in range(min(3, len(pdf_document))):  # Limit to first 3 pages
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                all_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
        
        pdf_document.close()
        return all_text.strip() if all_text.strip() else None
            
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return None

def extract_text_with_enhanced_ocr(image_bytes, file_type):
    """Enhanced OCR with preprocessing and PDF support"""
    try:
        if file_type == "application/pdf":
            result = extract_text_from_pdf(image_bytes)
            if result and len(result.strip()) > 3:
                return result
        else:
            # Handle images with OCR API
            result = extract_text_from_image_api(image_bytes)
            if result and len(result.strip()) > 3:
                return result
        return None
    except Exception as e:
        print(f"Enhanced OCR failed: {e}")
        return None

class ExtractionResult(BaseModel):
    raw_text: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    category: Optional[str] = None
    transaction_status: Optional[str] = None
    confidence: float = 0.0

class ErrorResponse(BaseModel):
    error: str
    detail: str

def extract_vendor_name(text: str) -> Optional[str]:
    """Extract vendor name with improved logic"""
    lines = text.split('\n')
    
    # Look for common business indicators
    business_indicators = ['restaurant', 'cafe', 'store', 'shop', 'company', 'inc', 'llc', 'corp', 'bank', 'pay', 'opay', 'paypal', 'stripe', 'hospital', 'clinic', 'university', 'college', 'hotel', 'gas station']
    
    # First, look for specific payment platforms and major companies
    known_companies = ['OPay', 'PayPal', 'Stripe', 'Square', 'Venmo', 'Cash App', 'Zelle', 'Amazon', 'Google', 'Microsoft', 'Apple', 'Netflix', 'Spotify', 'Uber', 'Lyft', 'Airbnb']
    for company in known_companies:
        if company.lower() in text.lower():
            return company
    
    # Look for business names in the first few lines
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 3 and not re.match(r'^\d+$', line):
            # Skip bullet points and special characters
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
            
            # Skip common non-business words
            skip_words = ['receipt', 'invoice', 'bill', 'statement', 'transaction', 'payment', 'total', 'amount', 'date', 'time']
            if any(skip_word in line.lower() for skip_word in skip_words):
                continue
            
            # Check if line contains business indicators
            if any(indicator in line.lower() for indicator in business_indicators):
                return line
            # Check if line looks like a business name (no excessive numbers, reasonable length)
            if not re.search(r'\d{4,}', line) and 3 <= len(line) <= 60:
                # Additional check: should contain letters
                if re.search(r'[a-zA-Z]', line):
                    return line
    
    return None

def extract_date(text: str) -> Optional[str]:
    """Extract date using various patterns with better accuracy"""
    date_patterns = [
        # Common date formats
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b',  # YYYY/MM/DD
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b',  # DD Mon YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})\b',  # Mon DD, YYYY
        # Ordinal date formats (1st, 2nd, 3rd, 4th, etc.)
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b',  # Mon 13th, 2025
        # More flexible patterns
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b',  # MM/DD/YYYY
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',  # MM/DD/YYYY
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})\b',  # MM/DD/YY
        # Look for date keywords
        r'(?:date|on|issued|created)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(?:date|on|issued|created)[:\s]*(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',
    ]
    
    # First try patterns with context
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                return '/'.join(matches[0])
            return matches[0]
    
    # If no context patterns work, try simple date patterns
    simple_patterns = [
        r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        r'\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b',
    ]
    
    for pattern in simple_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    return None

def extract_total_amount(text: str) -> Optional[float]:
    """Extract total amount using various patterns with better filtering"""
    # Look for currency patterns with better context
    currency_patterns = [
        # Business invoice patterns
        r'total[:\s]*\$?(\d+\.?\d{1,2})',  # Allow 1-2 decimal places
        r'amount[:\s]*\$?(\d+\.?\d{1,2})',
        r'sum[:\s]*\$?(\d+\.?\d{1,2})',
        r'grand\s*total[:\s]*\$?(\d+\.?\d{1,2})',
        r'subtotal[:\s]*\$?(\d+\.?\d{1,2})',
        r'amount\s*due[:\s]*\$?(\d+\.?\d{1,2})',
        r'balance[:\s]*\$?(\d+\.?\d{1,2})',
        r'outstanding[:\s]*\$?(\d+\.?\d{1,2})',
        r'payment\s*due[:\s]*\$?(\d+\.?\d{1,2})',
        r'net\s*amount[:\s]*\$?(\d+\.?\d{1,2})',
        r'final\s*amount[:\s]*\$?(\d+\.?\d{1,2})',
        # Currency symbol patterns
        r'\$(\d+\.?\d{1,2})',  # Dollar sign with 1-2 decimals
        r'(\d+\.?\d{1,2})\s*dollars?',
        # Korean Won patterns
        r'₩(\d+(?:,\d{3})*(?:\.\d{1,2})?)',  # Korean Won symbol with commas
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*won',
        # Other currency patterns
        r'€(\d+\.?\d{1,2})',  # Euro
        r'£(\d+\.?\d{1,2})',  # Pound
        r'¥(\d+\.?\d{1,2})',  # Yen
        r'₹(\d+\.?\d{1,2})',  # Indian Rupee
        r'₽(\d+\.?\d{1,2})',  # Russian Ruble
        # More flexible patterns
        r'total[:\s]*\$?(\d+\.?\d*)',  # Any decimal places
        r'amount[:\s]*\$?(\d+\.?\d*)',
        r'\$(\d+\.?\d*)',  # Any dollar amount
        r'₩(\d+\.?\d*)',  # Any Korean Won amount
    ]
    
    # Filter out common false positives
    false_positive_indicators = [
        'session id', 'order id', 'receipt id', 'transaction id',
        'phone number', 'zip code', 'postal code', 'area code',
        'reference', 'ref', 'tracking', 'confirmation'
    ]
    
    # Check if text contains false positive indicators
    text_lower = text.lower()
    has_false_positives = any(indicator in text_lower for indicator in false_positive_indicators)
    
    # If we detect potential false positives, be more strict
    if has_false_positives:
        # Only look for clearly marked totals
        strict_patterns = [
            r'total[:\s]*\$?(\d+\.?\d{2})',
            r'grand\s*total[:\s]*\$?(\d+\.?\d{2})',
            r'amount\s*due[:\s]*\$?(\d+\.?\d{2})',
        ]
        currency_patterns = strict_patterns
    
    for pattern in currency_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Remove commas from the matched amount
                amount_str = matches[0].replace(',', '')
                amount = float(amount_str)
                # Reasonable range for invoice amounts (including Korean Won)
                if 0.01 <= amount <= 1000000:  # Increased range for Korean Won
                    return amount
            except ValueError:
                continue
    
    # If no clear currency patterns found, look for the largest reasonable number
    if not has_false_positives:
        numbers = re.findall(r'\b(\d+\.?\d{2})\b', text)  # Only 2 decimal places
        if numbers:
            try:
                amounts = [float(num) for num in numbers if 0.01 <= float(num) <= 10000]
                if amounts:
                    return max(amounts)  # Return the largest reasonable amount
            except ValueError:
                pass
    
    return None

def categorize_invoice(text: str) -> str:
    """Comprehensive categorization based on keywords"""
    text_lower = text.lower()
    
    # Banking/Finance indicators (check first as they're most specific)
    if any(word in text_lower for word in ['bank', 'banking', 'transaction', 'payment', 'transfer', 'opay', 'paypal', 'stripe', 'recipient', 'sender', 'account', 'deposit', 'withdrawal', 'credit card', 'debit card']):
        return 'Banking & Finance'
    # Business/Professional documents
    elif any(word in text_lower for word in ['invoice', 'bill', 'statement', 'invoice number', 'bill number', 'due date', 'payment due', 'amount due', 'total due', 'outstanding', 'balance']):
        return 'Business Invoice'
    elif any(word in text_lower for word in ['contract', 'agreement', 'terms', 'conditions', 'signature', 'signed', 'legal', 'lawyer', 'attorney']):
        return 'Legal Document'
    elif any(word in text_lower for word in ['medical', 'doctor', 'hospital', 'clinic', 'pharmacy', 'prescription', 'medicine', 'health', 'insurance', 'patient']):
        return 'Healthcare'
    elif any(word in text_lower for word in ['education', 'school', 'university', 'college', 'tuition', 'student', 'course', 'training', 'certificate']):
        return 'Education'
    elif any(word in text_lower for word in ['government', 'tax', 'irs', 'revenue', 'license', 'permit', 'registration', 'official', 'public']):
        return 'Government'
    elif any(word in text_lower for word in ['insurance', 'policy', 'coverage', 'claim', 'premium', 'deductible']):
        return 'Insurance'
    elif any(word in text_lower for word in ['utility', 'electric', 'water', 'gas', 'internet', 'phone', 'cable', 'internet bill', 'electric bill']):
        return 'Utilities'
    elif any(word in text_lower for word in ['subscription', 'monthly', 'annual', 'recurring', 'membership', 'plan']):
        return 'Subscription'
    # Retail/Consumer
    elif any(word in text_lower for word in ['restaurant', 'food', 'dining', 'cafe', 'bar', 'coffee', 'meal']):
        return 'Food & Dining'
    elif any(word in text_lower for word in ['gas', 'fuel', 'petrol', 'station', 'gas station', 'fuel station']):
        return 'Transportation'
    elif any(word in text_lower for word in ['hotel', 'lodging', 'accommodation', 'booking', 'reservation', 'travel']):
        return 'Travel'
    elif any(word in text_lower for word in ['grocery', 'supermarket', 'store', 'market', 'shopping', 'retail']):
        return 'Groceries'
    elif any(word in text_lower for word in ['office', 'supplies', 'stationery', 'business', 'workplace']):
        return 'Office Supplies'
    elif any(word in text_lower for word in ['entertainment', 'movie', 'cinema', 'theater', 'concert', 'show', 'ticket']):
        return 'Entertainment'
    elif any(word in text_lower for word in ['shopping', 'retail', 'store', 'purchase', 'buy', 'sale']):
        return 'Shopping'
    else:
        return 'Other'

def detect_transaction_status(text: str) -> str:
    """Detect transaction status from text"""
    text_lower = text.lower()
    
    # Success indicators (more comprehensive)
    success_indicators = ['paid', 'completed', 'successful', 'approved', 'processed', 'confirmed', 'success', 'done', 'finished']
    if any(indicator in text_lower for indicator in success_indicators):
        return 'Success'
    
    # Pending indicators
    pending_indicators = ['pending', 'processing', 'in progress', 'waiting', 'queued', 'pending approval']
    if any(indicator in text_lower for indicator in pending_indicators):
        return 'Pending'
    
    # Failed indicators
    failed_indicators = ['failed', 'declined', 'rejected', 'error', 'cancelled', 'void', 'denied', 'unsuccessful']
    if any(indicator in text_lower for indicator in failed_indicators):
        return 'Failed'
    
    # Default to success if we have amount and date (likely a receipt)
    if extract_total_amount(text) and extract_date(text):
        return 'Success'
    
    return 'Unknown'

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
        
        if not raw_text or len(raw_text.strip()) < 3:
                raise HTTPException(
                status_code=400,
                detail="Could not extract text from the uploaded file. Please try a clearer image or PDF."
            )
        
        # Calculate confidence based on extraction success and text quality
        extracted_fields = 0
        total_fields = 4  # vendor, date, amount, category
        
        # Check what we can extract
        vendor = extract_vendor_name(raw_text)
        date = extract_date(raw_text)
        amount = extract_total_amount(raw_text)
        category = categorize_invoice(raw_text)
        
        if vendor: extracted_fields += 1
        if date: extracted_fields += 1
        if amount: extracted_fields += 1
        if category and category != "Other": extracted_fields += 1
        
        # Base confidence on extraction success
        extraction_confidence = (extracted_fields / total_fields) * 0.8
        
        # Add text quality factor
        text_quality = min(0.2, len(raw_text) / 2000)  # Up to 20% for text quality
        
        confidence = min(0.95, extraction_confidence + text_quality)
        confidence_scores = [confidence]
        
        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded file"
            )
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Get transaction status (not used in confidence calculation)
        transaction_status = detect_transaction_status(raw_text)
        
        return ExtractionResult(
            raw_text=raw_text.strip(),
            vendor=vendor,
            date=date,
            total_amount=total_amount,
            category=category,
            transaction_status=transaction_status,
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
