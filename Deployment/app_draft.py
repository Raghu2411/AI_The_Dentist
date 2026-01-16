import os, time
import json
import uuid
import re
from collections import deque
from flask import (
    Flask, request, jsonify, render_template, 
    session, send_file, Response, after_this_request
)
from flask_session import Session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import cv2
import numpy as np
from groq import Groq, GroqError


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
app.config["SESSION_FILE_DIR"] = os.path.join(config.BASE_DIR, 'flask_session')

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

# Initialize the session extension
Session(app)

# Configuration
UPLOADS_DIR = config.UPLOADS_DIR
RESULTS_DIR = config.RESULTS_DIR

MODEL_ID = 'qwen/qwen3-32b' 

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
    print("WARNING: No GROQ_API_KEY environment variables found. ")
    api_key_queue = deque()
    GROQ_CLIENT = None
else:
    print(f"Loaded {len(GROQ_API_KEYS)} API keys.")
    api_key_queue = deque(GROQ_API_KEYS)
    GROQ_CLIENT = Groq(api_key=api_key_queue[0])

# Pre-load All 5 Detection Models on Startup
print("--- Server is starting: Loading all detection models into memory... ---")
MODELS = utils_detection.load_all_models()
print("--- Model loading complete. Server is ready. ---")

# Get New Groq Client
def get_new_groq_client():
    """Rotates the global API key queue and returns a new client."""
    global GROQ_CLIENT, api_key_queue
    
    if not api_key_queue or len(api_key_queue) == 0:
        print("No API keys available.")
        return False

    api_key_queue.rotate(-1)
    new_key = api_key_queue[0]
    
    if len(GROQ_API_KEYS) > 0 and new_key == GROQ_API_KEYS[0]:
        print("All API keys have been tried and are likely exhausted.")
        
    print(f"\nRotating to new API key: ...{new_key[-4:]}")
    GROQ_CLIENT = Groq(api_key=new_key)
    return True

# === Main Application Routes ===

@app.route('/')
def index():
    """Serves the main chat/upload HTML page."""
    session.clear()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    The main endpoint for Detection and Report Generation.
    Receives an image and model choices, returns the full report.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image']
    selected_models = request.form.getlist('models')
    
    if not image_file or image_file.filename == '':
        return jsonify({"error": "No image selected."}), 400
    if not selected_models:
        return jsonify({"error": "No models selected."}), 400
    
    # Save Uploaded Image
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

    # Run Detection Pipeline
    all_predictions = []
    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"Warning: Selected model '{model_name}' not found in pre-loaded models.")
            continue
        
        model = MODELS[model_name]
        print(f"Running inference with: {model_name}")
        
        try:
            if model_name == 'yolov9':
                all_predictions.extend(utils_detection.run_yolo_inference(model, image_cv2))
            elif model_name == 'detectron2':
                all_predictions.extend(utils_detection.run_detectron2_inference(model, image_cv2))
            # elif model_name == 'mmdetection':
            #     all_predictions.extend(utils_detection.run_mmdetection_inference(model, image_cv2))
            elif model_name in ['faster_rcnn', 'retinanet']:
                all_predictions.extend(utils_detection.run_torchvision_inference(model, image_pil, model_name))
        except Exception as e:
            print(f"ERROR running model {model_name}: {e}")
            return jsonify({"error": f"Failed during inference with {model_name}. Is the model loaded correctly? Error: {e}"}), 500

    # Run Post-Processing Pipeline
    print("Running post-processing...")
    filtered_preds = utils_detection.filter_by_confidence(all_predictions)
    final_preds_list = utils_detection.apply_batched_nms(filtered_preds, config.NMS_IOU_THRESHOLD)
    structured_teeth, unassigned = utils_detection.assign_diseases_to_teeth(final_preds_list)
    
    final_structured_predictions = {
        "teeth": structured_teeth,
        "unassigned": unassigned
    }

    # Generate Caption & Report
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

    # Generate Predicted Image
    print("Generating prediction image...")
    predicted_filename = f"{unique_id}_predicted.jpg"
    predicted_image_path_fs = os.path.join(config.RESULTS_DIR, predicted_filename) # Filesystem path
    utils_visualization.draw_predictions_on_image(
        original_image_path,
        final_structured_predictions,
        predicted_image_path_fs
    )
    predicted_image_url = f"/static/results/{predicted_filename}"

    # Save to Session for Chat & Downloads
    print("Saving data to user session.")
    session['caption'] = caption_text
    session['report'] = report_text
    session['original_image_path'] = original_image_path 
    session['predicted_image_path'] = predicted_image_path_fs 
    session['predicted_image_url'] = predicted_image_url 

    return jsonify({
        "original_image_url": f"/uploads/{original_filename}", 
        "predicted_image_url": predicted_image_url,
        "caption": caption_text,
        "report": report_text
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the original uploaded image to the user."""
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_file(os.path.join(config.UPLOADS_DIR, filename))

# Download Option Endpoints

@app.route('/download_caption')
def download_caption():
    """Sends the caption text as a .txt file."""
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
    """Generates and sends the full PDF report."""
    report_text = session.get('report')
    original_img = session.get('original_image_path')
    predicted_img = session.get('predicted_image_path')

    if not all([report_text, original_img, predicted_img]):
        return "Report data not found in session. Please analyze an image first.", 404

    pdf_filename = f"{uuid.uuid4()}_report.pdf"
    pdf_save_path = os.path.join(config.RESULTS_DIR, pdf_filename)

    try:
        success = utils_generation.create_report_pdf(
            original_image_path=original_img,
            predicted_image_path=predicted_img,
            report_text=report_text,
            pdf_save_path=pdf_save_path
        )
        if not success:
            return jsonify({"error": "Failed to generate PDF."}), 500
    except Exception as e:
        return jsonify({"error": f"Failed during PDF generation: {e}"}), 500

    @after_this_request
    def cleanup(response):
        if os.path.exists(pdf_save_path):
            try:
                os.remove(pdf_save_path)
            except Exception as e:
                print(f"Error removing temporary file {pdf_save_path}: {e}")
        return response

    # cleanup funtion will auto run!
    return send_file(
        pdf_save_path,
        as_attachment=True,
        download_name='medical_report.pdf'
    )


# Chatbot (RAG) Endpoint

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the RAG chat logic."""
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
        {"role": "assistant", "content": "I have read your report. I am ready to help. What is your first question?"}
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
                        if "</think>" in full_response_buffer:
                            think_tag_found = True
                            clean_part = re.sub(r".*</think>", "", full_response_buffer, flags=re.DOTALL | re.IGNORECASE)
                            yield clean_part.lstrip()

            except GroqError as e:
                is_tpd_limit = e.status_code == 429 and "tpd" in str(e.message).lower()
                is_restricted = e.status_code == 400 and "organization_restricted" in str(e.message).lower()

                if is_tpd_limit or is_restricted:
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

if __name__ == '__main__':
    if not MODELS:
        print("--- Server failed to start: Models could not be loaded. ---")
    else:
        print("Flask server starting...")
        print("Open http://127.0.0.1:5000 in your browser.")
        app.run(debug=True, port=5000)