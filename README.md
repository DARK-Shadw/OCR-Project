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
5.  **For PDF processing:** Install Poppler. Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases).
    *   You will need to update the `poppler_path` in `app.py` to point to the `bin` directory of your Poppler installation (e.g., `r'C:\Program Files\poppler-0.68.0\bin'`)

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

### `POST /process_question_paper`

Processes an image or PDF of a question paper to extract questions and answers.

**Request:** `multipart/form-data` with `file` (a single image or PDF file).

**Example (curl for image):**

```bash
curl -X POST -F "file=@/path/to/your/question_paper.png" http://127.0.0.1:5000/process_question_paper
```

**Example (curl for PDF):**

```bash
curl -X POST -F "file=@/path/to/your/question_paper.pdf" http://127.0.0.1:5000/process_question_paper
```

### `GET /evaluate_answers`

Compares OCR extracted texts with the answers from the last processed question paper.

**Request:** None (GET request).

**Example (curl):**

```bash
curl -X GET http://127.0.0.1:5000/evaluate_answers
```

