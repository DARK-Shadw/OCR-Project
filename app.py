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
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
reader = easyocr.Reader(['en'])

# Global variable to store OCR extracted texts
ocr_extracted_texts = []
last_processed_question_paper_object = None

class QuestionPaper:
    def __init__(self, path=None):
        self.questions = []
        self.answers = []
        self.path = path

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

def improved_clean_and_parse_ocr_text(ocr_text):
    """
    Improved parsing with better answer extraction logic
    """
    # Remove special characters but keep important ones
    cleaned_text = re.sub(r'[|@~¥#$%^&*()_+=\[\]{}\\:";\'<>?,./]', ' ', ocr_text)
    
    # Split by newlines and filter out empty strings
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    
    individual_answers = []
    
    # Try to find numbered patterns first
    numbered_pattern = re.compile(r'(\d+)\s*[.)]\s*([^0-9]+?)(?=\d+\s*[.)]|$)', re.MULTILINE | re.DOTALL)
    matches = numbered_pattern.findall(cleaned_text)
    
    if matches:
        # If we found numbered patterns, use them
        for number, answer in matches:
            answer = answer.strip()
            if answer and len(answer) > 1:
                individual_answers.append(answer)
    else:
        # Fallback to line-by-line processing
        for line in lines:
            # Remove leading numbers and punctuation
            cleaned_line = re.sub(r'^\d+\s*[.)]\s*', '', line).strip()
            if cleaned_line and len(cleaned_line) > 1:
                individual_answers.append(cleaned_line)
    
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
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
                image_file.save(temp_image_file.name)
                temp_path = temp_image_file.name

            try:
                image_np = np.frombuffer(open(temp_path, 'rb').read(), np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                
                # Perform OCR
                result = reader.readtext(image)
                
                # Extract text from the result
                text = " ".join([item[1] for item in result])
                extracted_texts.append(text)
                ocr_extracted_texts.append(text)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
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
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
                image_file.save(temp_image_file.name)
                temp_path = temp_image_file.name

            try:
                with Image.open(temp_path) as image:
                    # Perform OCR using Tesseract
                    text = pytesseract.image_to_string(image)
                    extracted_texts.append(text.strip())
                    ocr_extracted_texts.append(text.strip())
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
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
        # Create Images directory if it doesn't exist
        images_dir = os.path.join(app.root_path, 'Images')
        os.makedirs(images_dir, exist_ok=True)
        
        if file.filename.lower().endswith('.pdf'):
            question_paper_filename = "question_paper.pdf"
            question_paper_path = os.path.join(images_dir, question_paper_filename)
            file.save(question_paper_path)
            
            # Initialize the global object with the path
            question_paper.path = question_paper_path
            
            # For PDF processing
            images_from_pdf = convert_from_path(question_paper_path, poppler_path=r'C:\Program Files\poppler\Library\bin')
            
            for page_image in images_from_pdf:
                text = pytesseract.image_to_string(page_image)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for i, line in enumerate(lines):
                    if i % 2 == 0:
                        question_paper.add_question(line)
                    else:
                        question_paper.add_answer(line)
        
        else:
            # Process as image
            question_paper_filename = "question_paper.png"
            question_paper_path = os.path.join(images_dir, question_paper_filename)
            file.save(question_paper_path)
            
            question_paper.path = question_paper_path
            
            image = Image.open(question_paper_path)
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

def gemini_evaluate_answer_sheet(question_paper_path, student_answer_path, questions, correct_answers, student_name=None):
    """
    Evaluate entire answer sheet using Gemini at once
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create the expected answers list for the prompt
        expected_answers_text = "\n".join([f"{i+1}. {answer}" for i, answer in enumerate(correct_answers)])
        
        prompt_text = f"""You are a teacher grading a test for children. 
        I will provide you with a question paper and a student's answer sheet.
        
        Expected correct answers:
        {expected_answers_text}
        
        Instructions:
        - Compare the student's handwritten answers with the expected answers above
        - Small spelling mistakes should be ignored and considered correct
        - If an answer has been crossed out or strikethrough, consider it incorrect
        - Be lenient with handwriting recognition issues
        - Look for answers by question numbers (1, 2, 3, etc.)
        
        Please evaluate ALL questions and respond in this EXACT JSON format:
        {{
            "evaluations": [
                {{"question_number": 1, "status": "Correct"}},
                {{"question_number": 2, "status": "Wrong"}},
                {{"question_number": 3, "status": "Missing"}},
                ...
            ]
        }}
        
        For each question, use ONLY one of these three status values:
        - "Correct" - if the student's answer matches the expected answer (allowing for minor spelling)
        - "Wrong" - if the student's answer is clearly different from the expected answer  
        - "Missing" - if no answer is visible for this question number
        
        Respond with ONLY the JSON format above, no other text."""

        # Handle PDF vs Image for question paper
        if question_paper_path.lower().endswith('.pdf'):
            # Convert PDF to images
            pdf_images = convert_from_path(question_paper_path, poppler_path=r'C:\Program Files\poppler\Library\bin')
            question_paper_img = pdf_images[0]  # Use first page
        else:
            question_paper_img = Image.open(question_paper_path)
        
        # Load student answer image
        student_answer_img = Image.open(student_answer_path)
        
        # Create content for the model
        content = [prompt_text, question_paper_img, student_answer_img]
        
        response = model.generate_content(content)
        result_text = response.text.strip()
        
        print(f"Gemini response: {result_text}")
        
        # Try to parse JSON response
        import json
        try:
            # Clean the response - sometimes Gemini adds markdown formatting
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            parsed_result = json.loads(result_text)
            return parsed_result["evaluations"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {result_text}")
            # Fallback - create default "Error" results
            return [{"question_number": i+1, "status": "Error"} for i in range(len(correct_answers))]
            
    except Exception as e:
        print(f"Error in Gemini evaluation: {str(e)}")
        # Return error status for all questions
        return [{"question_number": i+1, "status": "Error"} for i in range(len(correct_answers))]

@app.route('/evaluate_answers', methods=['POST'])
def evaluate_answers():
    global ocr_extracted_texts
    if 'student_answers' not in request.files:
        return jsonify({"error": "Missing student answers"}), 400

    student_answer_files = request.files.getlist('student_answers')
    student_name = request.form.get('student_name')

    # Retrieve the question paper object
    question_paper = last_processed_question_paper_object

    if last_processed_question_paper_object is None:
        return jsonify({'error': 'Question paper not found or processed yet'}), 404

    student_answer_paths = []
    try:
        # Save student answer files temporarily
        for student_answer_file in student_answer_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_ans_file:
                student_answer_file.save(temp_ans_file.name)
                student_answer_paths.append(temp_ans_file.name)

        # Traditional OCR evaluation (keep as fallback)
        ocr_results = []
        for student_answer_path in student_answer_paths:
            student_answer_image = Image.open(student_answer_path)
            student_answer_text = pytesseract.image_to_string(student_answer_image)
            student_individual_answers = improved_clean_and_parse_ocr_text(student_answer_text)

            for i, correct_answer in enumerate(question_paper.answers):
                if i < len(student_individual_answers):
                    student_ans = student_individual_answers[i]
                    best_match, score = find_best_match(student_ans, [correct_answer])
                    
                    status = "Wrong"
                    if best_match:
                        status = "Correct"
                    
                    ocr_results.append({
                        'question_number': i + 1,
                        'correct_answer': correct_answer,
                        'student_answer': student_ans,
                        'status': status,
                        'similarity_score': score
                    })
                else:
                    ocr_results.append({
                        'question_number': i + 1,
                        'correct_answer': correct_answer,
                        'student_answer': 'N/A',
                        'status': 'Missing',
                        'similarity_score': 0
                    })

        # Gemini evaluation (main evaluation method)
        gemini_results = []
        if question_paper.path and os.path.exists(question_paper.path):
            print("Starting Gemini evaluation...")
            
            # Process each student answer sheet once
            for idx, student_answer_path in enumerate(student_answer_paths):
                print(f"Evaluating answer sheet {idx + 1} with Gemini...")
                
                # Get evaluation for entire answer sheet at once
                sheet_evaluations = gemini_evaluate_answer_sheet(
                    question_paper.path, 
                    student_answer_path, 
                    question_paper.questions,
                    question_paper.answers,
                    student_name
                )
                
                # Process the results and add question details
                for eval_result in sheet_evaluations:
                    question_num = eval_result["question_number"]
                    if 1 <= question_num <= len(question_paper.questions):
                        gemini_results.append({
                            'question_number': question_num,
                            'question_text': question_paper.questions[question_num - 1],
                            'correct_answer': question_paper.answers[question_num - 1],
                            'status': eval_result["status"],
                            'answer_sheet': idx + 1  # Track which answer sheet this is from
                        })
            
            # Calculate summary
            correct_count = sum(1 for result in gemini_results if result['status'] == 'Correct')
            total_questions = len(gemini_results)
            score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
            
            final_results = {
                'student_name': student_name,
                'total_questions': len(question_paper.answers),
                'total_answer_sheets': len(student_answer_paths),
                'correct_answers': correct_count,
                'wrong_answers': sum(1 for result in gemini_results if result['status'] == 'Wrong'),
                'missing_answers': sum(1 for result in gemini_results if result['status'] == 'Missing'),
                'error_answers': sum(1 for result in gemini_results if result['status'] == 'Error'),
                'score_percentage': round(score_percentage, 2),
                'gemini_evaluation_results': gemini_results,
                'ocr_evaluation_results': ocr_results  # Keep as reference
            }
            
            return jsonify(final_results)
        else:
            return jsonify({
                'ocr_evaluation_results': ocr_results, 
                'gemini_evaluation_warning': 'Question paper file not found for Gemini evaluation.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary student answer files
        for path in student_answer_paths:
            if os.path.exists(path):
                os.unlink(path)

@app.route('/debug_parsing', methods=['GET'])
def debug_parsing():
    """
    Debug endpoint to see how OCR text is being parsed
    """
    if not ocr_extracted_texts:
        return jsonify({'error': 'No OCR extracted texts available.'}), 400
    
    debug_results = []
    
    for ocr_text in ocr_extracted_texts:
        parsed_answers = improved_clean_and_parse_ocr_text(ocr_text)
        debug_results.append({
            'original_ocr_text': ocr_text,
            'parsed_answers': parsed_answers
        })
    
    return jsonify({'debug_results': debug_results})

if __name__ == '__main__':
    app.run(debug=True)