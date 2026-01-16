import os, time
import json
import uuid
import re
from collections import deque
from flask import (
    Flask, request, jsonify, render_template, 
    session, send_file, Response, after_this_request, redirect
)
from flask_session import Session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import cv2
import numpy as np
from groq import Groq, GroqError

import firebase_admin
from firebase_admin import credentials, firestore, storage

import config_master as config
import utils_detection
import utils_generation
import utils_visualization


# App Initialization
app = Flask(__name__)
CORS(app) 
app.secret_key = os.urandom(24)

# Configure server-side sessions
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_FILE_DIR"] = os.path.join(config.BASE_DIR, 'flask_session')

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

Session(app)

# Configuration
UPLOADS_DIR = config.UPLOADS_DIR
RESULTS_DIR = config.RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = 'qwen/qwen3-32b'

#  Groq API Key Rotation 
GROQ_API_KEYS = []
i = 1
while True:
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key:
        GROQ_API_KEYS.append(key)
        i += 1
    else:
        break

if not GROQ_API_KEYS:
    main_key = os.getenv("GROQ_API_KEY")
    if main_key:
        GROQ_API_KEYS.append(main_key)
        print("Loaded 1 GROQ_API_KEY from environment.")
    else:
        print("WARNING: No GROQ_API_KEY environment variables found. ")
    
if not GROQ_API_KEYS:
    api_key_queue = deque()
    GROQ_CLIENT = None
else:
    print(f"Loaded {len(GROQ_API_KEYS)} API keys.")
    api_key_queue = deque(GROQ_API_KEYS)
    GROQ_CLIENT = Groq(api_key=api_key_queue[0])

#  Pre-load All Detection Models on Startup 
print("Server is starting: Loading all detection models into memory...")
MODELS = utils_detection.load_all_models()
if not MODELS:
    print("CRITICAL WARNING: NO MODELS WERE LOADED.")
else:
    print(f"Models loaded successfully: {list(MODELS.keys())}")

db = None
bucket = None
try:
    firebase_secret_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    
    if not firebase_secret_json:
        raise FileNotFoundError 

    # Convert the JSON string back into a Python dictionary
    firebase_creds_dict = json.loads(firebase_secret_json)
    cred = credentials.Certificate(firebase_creds_dict)
    BUCKET_NAME = "ai-the-dentist.firebasestorage.app" 
    
    firebase_admin.initialize_app(cred, {
        'storageBucket': BUCKET_NAME
    })
    
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase (from Secret) initialized successfully. Feedback feature is ON.")

except FileNotFoundError:
    print("WARNING: 'FIREBASE_SERVICE_ACCOUNT' secret not set. Feedback feature will be DISABLED.")
except ValueError as e:
    if "The default Firebase app already exists" in str(e):
        print("Firebase already initialized.")
        if not db: db = firestore.client()
        if not bucket: bucket = storage.bucket()
    else:
        print(f"--- ERROR: Failed to initialize Firebase: {e} ---")
except Exception as e:
    print(f"--- ERROR: Failed to initialize Firebase: {e} ---")


# FB-STORAGE: Helper function to upload files
def upload_to_storage(file_path, destination_blob_name, bucket_obj):
    """Uploads a file to Firebase Storage and makes it public."""
    if not bucket_obj:
        return None
    try:
        # Create a blob and upload the file
        blob = bucket_obj.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        blob.make_public()
        
        # Return the public URL
        return blob.public_url
    except Exception as e:
        print(f"Error uploading {destination_blob_name} to Firebase Storage: {e}")
        return None

#  Groq Helper Functions
def get_new_groq_client():
    """Rotates the global API key queue and returns a new client."""
    global GROQ_CLIENT, api_key_queue
    
    if not api_key_queue or len(api_key_queue) == 0:
        print("No API keys available.")
        return False

    api_key_queue.rotate(-1)
    new_key = api_key_queue[0]
    
    if len(GROQ_API_KEYS) > 1 and new_key == GROQ_API_KEYS[0]:
        print("All API keys have been tried and are likely exhausted.")
        
    print(f"\nRotating to new API key: ...{new_key[-4:]}")
    GROQ_CLIENT = Groq(api_key=new_key)
    return True

def get_chat_response_non_streaming(msgs):
    """
    Gets a non-streaming chat response from Groq, with one retry on key failure.
    Used by the stateless API.
    """
    global GROQ_CLIENT, api_key_queue
    if not GROQ_CLIENT:
        return "ERROR: Groq API client is not initialized."
    
    try:
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=msgs,
            model=MODEL_ID,
            temperature=0.7,
            max_tokens=2048,
            stream=False
        )
        return utils_generation.clean_llm_output(chat_completion.choices[0].message.content)
    
    except GroqError as e:
        is_tpd_limit = e.status_code == 429 and "tpd" in str(e.message).lower()
        is_restricted = e.status_code == 400 and "organization_restricted" in str(e.message).lower()

        if (is_tpd_limit or is_restricted) and len(GROQ_API_KEYS) > 1:
            print(f"API key error in non-streaming chat: {e.message}. Rotating key.")
            if not get_new_groq_client():
                return "ERROR: All API keys have hit their daily token limits or have been restricted."
            
            try:
                chat_completion = GROQ_CLIENT.chat.completions.create(
                    messages=msgs,
                    model=MODEL_ID,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=False
                )
                return utils_generation.clean_llm_output(chat_completion.choices[0].message.content)
            except Exception as retry_e:
                return f"ERROR: API call failed on retry: {str(retry_e)}"
        
        else:
            return f"ERROR: An API error occurred: {str(e)}"
    except Exception as e:
        return f"ERROR: A non-API error occurred: {str(e)}"

# WEB APPLICATION ROUTES (FOR BROWSER)
@app.route('/')
def index():
    """Serves the main chat/upload HTML page."""
    session.clear()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    The main endpoint for Detection and Report Generation.
    Receives an image and model choices.
    
    - FOR (WEB APP): Saves to session and returns JSON.
    - FOR (MOBILE API): Saves to disk and returns JSON with job_id and URLs.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image']
    selected_models = request.form.getlist('models') or request.form.getlist('models[]')
    
    if not image_file or image_file.filename == '':
        return jsonify({"error": "No image selected."}), 400
    if not selected_models:
        return jsonify({"error": "No models selected."}), 400
    
    # Save local temporary copy of uploaded image
    filename = secure_filename(image_file.filename)
    unique_id = str(uuid.uuid4())
    original_filename = f"{unique_id}_{filename}"
    original_image_path = os.path.join(config.UPLOADS_DIR, original_filename)
    image_file.save(original_image_path)
    
    try:
        image_pil = PILImage.open(original_image_path).convert('RGB')
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400
    
    if not MODELS:
        return jsonify({"error": "Models are not loaded. Server configuration error."}), 500

    # Run AI Pipeline
    all_predictions = []
    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"Warning: Selected model '{model_name}' not found in pre-loaded models.")
            continue
        
        model = MODELS[model_name]
        print(f"Running inference with: {model_name}")
        
        try:
            if model_name in ['yolov9', 'yolov11']: 
                all_predictions.extend(utils_detection.run_yolo_inference(model, image_cv2))
            elif model_name == 'detectron2':
                all_predictions.extend(utils_detection.run_detectron2_inference(model, image_cv2))
            elif model_name in ['faster_rcnn', 'retinanet']:
                all_predictions.extend(utils_detection.run_torchvision_inference(model, image_pil, model_name))
        except Exception as e:
            print(f"ERROR running model {model_name}: {e}")
            return jsonify({"error": f"Failed during inference with {model_name}: {e}"}), 500

    # Run Post-Processing Pipeline
    print("Running post-processing...")
    filtered_preds = utils_detection.filter_by_confidence(all_predictions)
    final_preds_list = utils_detection.apply_batched_nms(filtered_preds, config.NMS_IOU_THRESHOLD)
    structured_teeth, unassigned = utils_detection.assign_diseases_to_teeth(final_preds_list)
    
    final_structured_predictions = {
        "teeth": structured_teeth,
        "unassigned": unassigned
    }

    # Generate Reports 
    print("Generating grounding caption...")
    caption_text = utils_generation.create_grounding_caption(final_structured_predictions)

    print("Generating medical report via Groq...")
    report_text = "ERROR: No API keys available."
    if GROQ_CLIENT:
        report_text = utils_generation.get_groq_report_response(
            GROQ_CLIENT, caption_text, MODEL_ID 
        )

    if report_text.startswith("ERROR:"):
        return jsonify({"error": f"Failed to generate report: {report_text}"}), 500

    # Create Local Temporary Files (for chat and uploads)
    print("Generating local prediction image and PDF...")
    predicted_filename = f"{unique_id}_predicted.jpg"
    pdf_filename = f"{unique_id}_report.pdf"
    report_filename = f"{unique_id}_report.txt"
    caption_filename = f"{unique_id}_caption.txt"
    
    predicted_image_path_fs = os.path.join(config.RESULTS_DIR, predicted_filename)
    pdf_save_path = os.path.join(config.RESULTS_DIR, pdf_filename)
    report_save_path = os.path.join(config.RESULTS_DIR, report_filename)
    caption_save_path = os.path.join(config.RESULTS_DIR, caption_filename)
    
    utils_visualization.draw_predictions_on_image(
        original_image_path,
        final_structured_predictions,
        predicted_image_path_fs
    )
    utils_generation.create_report_pdf(
        original_image_path=original_image_path,
        predicted_image_path=predicted_image_path_fs,
        report_text=report_text,
        pdf_save_path=pdf_save_path
    )
    # Create local text files for chat API
    try:
        with open(caption_save_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        with open(report_save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    except Exception as e:
        print(f"Error saving local chat context files: {e}")

    # FB-STORAGE: Upload files and get permanent URLs
    perm_original_url = None
    perm_predicted_url = None
    perm_pdf_url = None

    if bucket:
        print("Uploading files to Firebase Storage for permanent URLs...")
        fb_original_path = f"feedback_files/{unique_id}/original.jpg"
        fb_predicted_path = f"feedback_files/{unique_id}/predicted.jpg"
        fb_pdf_path = f"feedback_files/{unique_id}/report.pdf"
        
        perm_original_url = upload_to_storage(original_image_path, fb_original_path, bucket)
        perm_predicted_url = upload_to_storage(predicted_image_path_fs, fb_predicted_path, bucket)
        perm_pdf_url = upload_to_storage(pdf_save_path, fb_pdf_path, bucket)
    
    if not perm_pdf_url:
        print("Firebase Storage upload failed or is disabled. Using ephemeral local URLs.")
        base_url = request.host_url.rstrip('/')
        perm_original_url = f"{base_url}/uploads/{original_filename}"
        perm_predicted_url = f"{base_url}/static/results/{predicted_filename}"
        perm_pdf_url = f"{base_url}/static/results/{pdf_filename}"
    
    # Save permanent URLs to session for website
    print("Saving data to user session for web app.")
    session['job_id'] = unique_id
    session['caption'] = caption_text
    session['report'] = report_text
    session['pdf_url'] = perm_pdf_url
    session['original_image_url'] = perm_original_url
    session['predicted_image_url'] = perm_predicted_url

    return jsonify({
        # (Web App) Fields - session-based URLs for fast display
        "original_image_url": f"/uploads/{original_filename}", 
        "predicted_image_url": f"/static/results/{predicted_filename}",
        "caption": caption_text,
        "report": report_text,
        
        # (Mobile API) Fields - the permanent Firebase URLs
        "job_id": unique_id,
        "api_urls": {
            "pdf_report": perm_pdf_url,
            "predicted_image": perm_predicted_url,
            "original_image": perm_original_url,
            "medical_report_text": report_text
        }
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the original uploaded image to the user."""
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_file(os.path.join(config.UPLOADS_DIR, filename))

@app.route('/static/results/<filename>')
def results_file(filename):
    """Serves the predicted image/pdf to the user."""
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_file(os.path.join(config.RESULTS_DIR, filename))


@app.route('/download_caption')
def download_caption():
    """(Web App) Sends the caption text as a .txt file from session."""
    caption = session.get('caption')
    if not caption:
        return "No caption found in session.", 404
    
    return Response(
        caption,
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=grounding_caption.txt"}
    )

@app.route('/download_pdf')
def download_pdf():
    """(Web App) Creates and sends the full PDF report from session."""
    # Redirect to the permanent URL
    pdf_url = session.get('pdf_url')
    if not pdf_url:
        return "Report data not found in session. Please analyze an image first.", 404

    return redirect(pdf_url)

@app.route('/chat', methods=['POST'])
def chat():
    """
    (Web App) Handles the RAG chat logic.
    Uses session for context and streams the response.
    """
    global GROQ_CLIENT, api_key_queue
    
    data = request.json
    question = data.get('question')
    history = data.get('history', [])
    
    if not GROQ_CLIENT:
         return jsonify({"error": "Groq API client is not initialized."}), 500

    # RAG LOGIC: Get context from session
    caption = session.get('caption')
    report = session.get('report')
    
    if not caption or not report:
        return jsonify({"error": "No report context found in session. Please analyze an image first."}), 404

    report_context = (
        f"--- PATIENT'S GROUNDING CAPTION (THE FACTS) ---\n{caption}\n"
        f"--- PATIENT'S MEDICAL REPORT (THE SUMMARY) ---\n{report}"
    )

    # Construct the prompt
    messages = [
        {"role": "system", "content": config.CHATBOT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is my patient report context:\n\n{report_context}\n\nPlease use this information to answer my questions."},
        {"role": "assistant", "content": "Hello, I have your dental report here and can help answer any questions you have about it."}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    
    # Call Groq API (with streaming and key rotation)
    try:
        def get_response(msgs):
            global GROQ_CLIENT, api_key_queue
            
            full_response_buffer = ""
            think_tag_found = False
            
            try:
                stream = GROQ_CLIENT.chat.completions.create(
                    messages=msgs,
                    model=MODEL_ID,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True
                )
                
                for chunk in stream:
                    chunk_content = chunk.choices[0].delta.content or ""
                    
                    if think_tag_found:
                        yield chunk_content
                    else:
                        full_response_buffer += chunk_content
                        # Clean the <think> tag
                        if "</think>" in full_response_buffer:
                            think_tag_found = True
                            clean_part = re.sub(r".*</think>", "", full_response_buffer, flags=re.DOTALL | re.IGNORECASE)
                            yield clean_part.lstrip()

            except GroqError as e:
                is_tpd_limit = e.status_code == 429 and "tpd" in str(e.message).lower()
                is_restricted = e.status_code == 400 and "organization_restricted" in str(e.message).lower()

                if (is_tpd_limit or is_restricted) and len(GROQ_API_KEYS) > 1:
                    if is_tpd_limit:
                        print(f"Key ...{api_key_queue[0][-4:]} exhausted (TPD). Rotating key.")
                    if is_restricted:
                        print(f"Key ...{api_key_queue[0][-4:]} has been BANNED (Organization Restricted). Rotating key.")
                    
                    if not get_new_groq_client():
                        yield "ERROR: All API keys have hit their daily token limits or have been restricted. Please try again tomorrow or contact support."
                        return
                    yield "ERROR: The server is busy or your key is exhausted. Retrying with a new key..."
                
                elif e.status_code == 429:
                    print("  [Info] Per-minute limit hit. Sleeping for 10 seconds...")
                    time.sleep(10)
                    yield "ERROR: Per-minute limit hit. Please re-send your message."
                else:
                    yield f"ERROR: An API error occurred: {str(e)}"
            except Exception as e:
                yield f"ERROR: A non-API error occurred: {str(e)}"

        return Response(get_response(messages), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# MOBILE APPLICATION API ROUTES (STATELESS)

@app.route('/api/v1/chat', methods=['POST'])
def api_chat():
    """
    (Mobile API) Handles the RAG chat logic for the stateless mobile app API.
    Receives a job_id and query, returns a single JSON response.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400
    
    job_id = data.get('job_id')
    query = data.get('query')
    history = data.get('history', []) # Expects [{"role": "user", "content": "..."}, ...]

    if not job_id or not query:
        return jsonify({"error": "Request must contain 'job_id' and 'query'"}), 400
    
    caption_path = os.path.join(config.RESULTS_DIR, f"{job_id}_caption.txt")
    report_path = os.path.join(config.RESULTS_DIR, f"{job_id}_report.txt")

    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption_context = f.read()
        with open(report_path, 'r', encoding='utf-8') as f:
            report_context = f.read()
    except FileNotFoundError:
        return jsonify({"error": "Invalid job_id or session expired. Please re-analyze your image."}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read context files: {e}"}), 500
    
    report_context = (
        f"--- PATIENT'S GROUNDING CAPTION (THE FACTS) ---\n{caption_context}\n"
        f"--- PATIENT'S MEDICAL REPORT (THE SUMMARY) ---\n{report_context}"
    )

    #  Construct the prompt 
    messages = [
        {"role": "system", "content": config.CHATBOT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is my patient report context:\n\n{report_context}\n\nPlease use this information to answer my questions."},
        {"role":"assistant", "content": "Hello, I have your dental report here and can help answer any questions you have about it."}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    #  Call Groq API (non-streaming)
    response_text = get_chat_response_non_streaming(messages)
    
    if response_text.startswith("ERROR:"):
        return jsonify({"error": response_text}), 500
    
    return jsonify({"response": response_text})

# FEEDBACK API ENDPOINT
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Receives feedback from either the website (session-based)
    or the mobile app (stateless, job_id-based).
    Saves the 9 requested fields to Firestore.
    """
    if not db:
        return jsonify({"error": "Feedback system is not configured on the server."}), 503
    
    data = request.json
    source = data.get('source')
    
    try:
        if source == 'website':
            # For the website, get data from the form and the session
            name = data.get('name')
            feedback_text = data.get('feedback_text')
            
            # Get the permanent URLs from the session
            job_id = session.get('job_id')
            medical_report = session.get('report')
            pdf_url = session.get('pdf_url')
            original_image_url = session.get('original_image_url')
            predicted_image_url = session.get('predicted_image_url')

            if not all([job_id, name, feedback_text, medical_report, pdf_url, original_image_url, predicted_image_url]):
                return jsonify({"error": "Session data incomplete or expired. Please analyze an image again."}), 400

        elif source == 'mobile_api': # For Mobile
            job_id = data.get('job_id')
            name = data.get('name')
            feedback_text = data.get('feedback_text')
            medical_report = data.get('medical_report')
            pdf_url = data.get('pdf_url')
            original_image_url = data.get('original_image_url')
            predicted_image_url = data.get('predicted_image_url')

            if not all([job_id, name, feedback_text, medical_report, pdf_url, original_image_url, predicted_image_url]):
                return jsonify({"error": "Missing required fields. Mobile must send all job data."}), 400
        
        else:
            return jsonify({"error": "Invalid or missing 'source' (must be 'website' or 'mobile_api')"}), 400

        feedback_doc = {
            'job_id': job_id,
            'name': name,
            'feedback_text': feedback_text,
            'medical_report': medical_report,
            'pdf_url': pdf_url,
            'original_image_url': original_image_url,
            'predicted_image_url': predicted_image_url,
            'source': source,
            'timestamp': firestore.SERVER_TIMESTAMP 
        }

        # Save to Firestore (collection "feedback")
        db.collection('feedback').add(feedback_doc)

        return jsonify({"success": True, "message": "Feedback submitted successfully."}), 201

    except Exception as e:
        print(f"Error in /feedback: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# APP RUNNER

if __name__ == '__main__':
    if not MODELS:
        print("--- Server failed to start: Models could not be loaded. ---")
    else:
        print("Flask server starting...")
        # For HF Spaces, gunicorn will be used. This is for local testing.
        port = int(os.environ.get("PORT", 7860))
        print(f"Open http://127.0.0.1:{port} in your browser.")
        app.run(host="0.0.0.0", port=port, debug=True)


# Mobile Detection API - https://tym24-ai-the-dentist.hf.space/analyze
# Key  	    Value            [Body tab and select form-data]
# image	    (select a file)
# models	yolov9
# models	detectron2

# Mobile ChatBot API - https://tym24-ai-the-dentist.hf.space/api/v1/chat
# Format
# {
#   "job_id": "260f4809-e97d-4b0c-b354-eb21a1cbcb63",
#   "query": "What is the main summary of my report?",
#   "history": []
# }


# new feedback endpoint

# API: https://tym24-ai-the-dentist.hf.space/feedback
# The mobile app must get the name and feedback_text from the user.

# JSON to be sent to the database (Firebase FireStore)
# {
#   "source": "mobile_api",
#   "name": "Jane Doe",
#   "feedback_text": "The prediction for caries on tooth #37 was correct, but it missed an obvious one on #38.",
#   "job_id": "a9b8c7d6-e5f4-a3b2-c1d0-e9f8a7b6c5d4",
#   "medical_report": "This is the full text of the medical report...",
#   "pdf_url": "https://[...].hf.space/static/results/a9b8..._report.pdf",
#   "original_image_url": "https://[...].hf.space/uploads/a9b8..._my_image.jpg",
#   "predicted_image_url": "https://[...].hf.space/static/results/a9b8..._predicted.jpg"
# }

# FYI: Below fields are saved in the database in this order.
# job_id, name , feedback_text, medical_report, pdf_url, original_image_url, predicted_image_url, source and timestamp