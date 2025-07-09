# OCR Backend

Backend API for OCR on handwritten images.

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
2.  Activate the environment:
    *   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **For Tesseract OCR:** Install Tesseract on your system. Download from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
    *   If `pytesseract` can't find Tesseract, you might need to set the path in `app.py`:
        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```

## Run

```bash
python app.py
```

API will be at `http://127.0.0.1:5000`.

## API Endpoints

### `POST /easyocr`

Uses EasyOCR to extract text from images.

**Request:** `multipart/form-data` with `images` (one or more image files).

**Example (curl):**

```bash
curl -X POST -F "images=@/path/to/your/image1.png" http://127.0.0.1:5000/easyocr
```

### `POST /tesseract`

Uses Tesseract OCR to extract text from images.

**Request:** `multipart/form-data` with `images` (one or more image files).

**Example (curl):**

```bash
curl -X POST -F "images=@/path/to/your/image1.png" http://127.0.0.1:5000/tesseract
```
