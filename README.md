# Intelligent Scanner Backend

A FastAPI-based backend service for extracting structured data from invoices and receipts using OCR.

## Features

- **OCR Processing**: Uses PaddleOCR for text extraction from images and PDFs
- **Smart Parsing**: Automatically extracts vendor, date, total amount, and category
- **RESTful API**: Clean FastAPI endpoints with proper error handling
- **CORS Support**: Configured for frontend communication

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /extract` - Upload file and extract data
- `GET /health` - Detailed health status

## Usage

Upload an image or PDF file to `/extract` endpoint to get structured data:

```json
{
  "raw_text": "Extracted text from OCR...",
  "vendor": "Restaurant Name",
  "date": "12/25/2023",
  "total_amount": 45.67,
  "category": "Food & Dining",
  "confidence": 0.95
}
```
