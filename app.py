import easyocr
from pdf2image import convert_from_path, convert_from_bytes
from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract
from PIL import Image
import easyocr
import os
import tempfile
import difflib
import re
from fuzzywuzzy import fuzz

app = Flask(__name__)
reader = easyocr.Reader(['en'])

# Global variable to store OCR extracted texts
ocr_extracted_texts = []
last_processed_question_paper_object = None

class QuestionPaper:
    def __init__(self):
        self.questions = []
        self.answers = []

    def clean_answers(self):
        # Remove unwanted patterns from answers
        unwanted_patterns = [
            "Time: 15 MinutesMarks: 20",
            "Time: 15 Minutes Marks: 20"
        ]
        self.answers = [answer for answer in self.answers if answer not in unwanted_patterns]
    
    def add_question(self, question_text):
        self.questions.append(question_text)
    
    def add_answer(self, answer_text):
        self.answers.append(answer_text)
    
    def to_dict(self):
        return {
            'questions': self.questions,
            'answers': self.answers
        }

def clean_and_parse_ocr_text(ocr_text):
    """
    Parse OCR text to extract individual answers
    """
    # Remove special characters and clean the text
    cleaned_text = re.sub(r'[|@~¥#$%^&*()_+=\[\]{}\\:";\'<>?,./]', ' ', ocr_text)
    
    # Split by newlines and filter out empty strings
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    
    individual_answers = []
    
    for line in lines:
        # Split by numbers at the beginning (like "1.", "2.", etc.)
        parts = re.split(r'\d+\.?\s*', line)
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:  # Filter out single characters and empty strings
                # Further split by multiple spaces
                words = part.split()
                for word in words:
                    word = word.strip()
                    if len(word) > 2:  # Only consider words with more than 2 characters
                        individual_answers.append(word)
    
    return individual_answers

def find_best_match(student_answer, correct_answers, threshold=0.6):
    """
    Find the best matching correct answer for a student answer
    """
    best_score = 0
    best_match = None
    
    for correct_answer in correct_answers:
        # Use multiple similarity metrics
        ratio_score = difflib.SequenceMatcher(None, student_answer.lower(), correct_answer.lower()).ratio()
        fuzzy_score = fuzz.ratio(student_answer.lower(), correct_answer.lower()) / 100.0
        partial_score = fuzz.partial_ratio(student_answer.lower(), correct_answer.lower()) / 100.0
        
        # Take the maximum of all scores
        combined_score = max(ratio_score, fuzzy_score, partial_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = correct_answer
    
    # Only return match if it meets the threshold
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

@app.route('/easyocr', methods=['POST'])
def easyocr_image():
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    images = request.files.getlist('images')
    extracted_texts = []

    for image_file in images:
        try:
            # Read the image file
            image_np = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Perform OCR
            result = reader.readtext(image)
            
            # Extract text from the result
            text = " ".join([item[1] for item in result])
            extracted_texts.append(text)
            ocr_extracted_texts.append(text)
        except Exception as e:
            extracted_texts.append(f"Error processing image with EasyOCR: {str(e)}")

    return jsonify({'extracted_texts': extracted_texts})

@app.route('/tesseract', methods=['POST'])
def tesseract_image():
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    images = request.files.getlist('images')
    extracted_texts = []

    for image_file in images:
        try:
            # Read the image file and convert to PIL Image
            image = Image.open(image_file.stream)

            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(image)
            extracted_texts.append(text.strip())
            ocr_extracted_texts.append(text.strip())
        except Exception as e:
            extracted_texts.append(f"Error processing image with Tesseract: {str(e)}")

    return jsonify({'extracted_texts': extracted_texts})

@app.route('/process_question_paper', methods=['POST'])
def process_question_paper():
    global last_processed_question_paper_object
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    question_paper = QuestionPaper()
    
    try:
        if file.filename.lower().endswith('.pdf'):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # For PDF processing
                images_from_pdf = convert_from_path(temp_path, poppler_path=r'C:\Program Files\poppler\Library\bin')
                
                for page_image in images_from_pdf:
                    text = pytesseract.image_to_string(page_image)
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    for i, line in enumerate(lines):
                        if i % 2 == 0:
                            question_paper.add_question(line)
                        else:
                            question_paper.add_answer(line)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        else:
            # Process as image
            image = Image.open(file.stream)
            
            text = pytesseract.image_to_string(image)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for i, line in enumerate(lines):
                if i % 2 == 0:
                    question_paper.add_question(line)
                else:
                    question_paper.add_answer(line)
        
        # Clean the answers
        question_paper.clean_answers()
        
        # Store the processed question paper globally
        last_processed_question_paper_object = question_paper
        
        return jsonify(question_paper.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_answers', methods=['GET'])
def evaluate_answers():
    if not ocr_extracted_texts:
        return jsonify({'error': 'No OCR extracted texts available. Please process images first.'}), 400

    if last_processed_question_paper_object is None:
        return jsonify({'error': 'Question paper not processed. Please process a question paper first.'}), 400

    question_paper_answers = last_processed_question_paper_object.answers
    
    if not question_paper_answers:
        return jsonify({'error': 'No correct answers found in question paper.'}), 400

    evaluation_results = []

    for ocr_text in ocr_extracted_texts:
        # Parse the OCR text to extract individual answers
        student_answers = clean_and_parse_ocr_text(ocr_text)
        
        answer_evaluations = []
        matched_correct_answers = set()  # To avoid double matching
        
        for student_answer in student_answers:
            # Find available correct answers (not already matched)
            available_correct_answers = [ans for ans in question_paper_answers if ans not in matched_correct_answers]
            
            if available_correct_answers:
                best_match, score = find_best_match(student_answer, available_correct_answers)
                
                if best_match:
                    matched_correct_answers.add(best_match)
                    status = "Correct" if score >= 0.8 else "Partially Correct" if score >= 0.6 else "Incorrect"
                else:
                    status = "No Match"
                
                answer_evaluations.append({
                    'student_answer': student_answer,
                    'matched_correct_answer': best_match,
                    'similarity_score': round(score, 3),
                    'status': status
                })
        
        # Calculate overall statistics
        total_answers = len(answer_evaluations)
        correct_count = len([eval for eval in answer_evaluations if eval['status'] == 'Correct'])
        partially_correct_count = len([eval for eval in answer_evaluations if eval['status'] == 'Partially Correct'])
        
        evaluation_results.append({
            'raw_ocr_text': ocr_text,
            'parsed_student_answers': student_answers,
            'individual_evaluations': answer_evaluations,
            'summary': {
                'total_answers_found': total_answers,
                'correct_answers': correct_count,
                'partially_correct_answers': partially_correct_count,
                'incorrect_answers': total_answers - correct_count - partially_correct_count,
                'accuracy_percentage': round((correct_count / total_answers * 100) if total_answers > 0 else 0, 2)
            }
        })

    return jsonify({'evaluation_results': evaluation_results})

@app.route('/debug_parsing', methods=['GET'])
def debug_parsing():
    """
    Debug endpoint to see how OCR text is being parsed
    """
    if not ocr_extracted_texts:
        return jsonify({'error': 'No OCR extracted texts available.'}), 400
    
    debug_results = []
    
    for ocr_text in ocr_extracted_texts:
        parsed_answers = clean_and_parse_ocr_text(ocr_text)
        debug_results.append({
            'original_ocr_text': ocr_text,
            'parsed_answers': parsed_answers
        })
    
    return jsonify({'debug_results': debug_results})

if __name__ == '__main__':
    app.run(debug=True)