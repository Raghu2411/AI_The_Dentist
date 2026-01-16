import os
import re
import json
import time
from flask import Flask, request, jsonify, render_template, Response
from groq import Groq, GroqError
from collections import deque
from chatbot_prompt import CHATBOT_SYSTEM_PROMPT

app = Flask(__name__)

REPORTS_DIR = 'C:/Users/USER/Desktop/OPG-segmentation/Part 2/final/Dental_Dissertation/ChatBot Pipeline/medical_reports'
MODEL_ID = 'qwen/qwen3-32b'
# 9_output_1-2-392-200036-9125-4-0-537872757-215144680-3503410942_cropped_jpg.rf.1071be642fe6265fd51996fc7aa5cd12_report.pdf
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
    print("WARNING: No GROQ_API_KEY environment variables found. API calls will fail.")
    
api_key_queue = deque(GROQ_API_KEYS)
client = Groq(api_key=api_key_queue[0]) if GROQ_API_KEYS else None

# Chat Page
@app.route('/')
def index():
    return render_template('index.html')

# RAG API Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat logic."""
    global client, api_key_queue
    
    data = request.json
    question = data.get('question')
    history = data.get('history', [])
    
    patient_pdf_name = data.get('patient_id') 

    if not patient_pdf_name:
        return jsonify({"error": "No Patient ID provided."}), 400
    if not client:
         return jsonify({"error": "Groq API client is not initialized. Did you set your API keys?"}), 500

    match = re.match(r'^\d+_(.*)_report\.pdf$', patient_pdf_name)
    if not match:
        return jsonify({"error": f"Invalid Patient ID format. Expected 'index_..._report.pdf', got '{patient_pdf_name}'"}), 400
    
    patient_file_base = match.group(1) 

    try:
        caption_path = os.path.join(REPORTS_DIR, f"{patient_file_base}_input_caption.txt")
        report_path = os.path.join(REPORTS_DIR, f"{patient_file_base}_medical_report.txt")
        
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption_content = f.read()
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
        report_context = (
            "--- PATIENT'S GROUNDING CAPTION (THE FACTS) ---\n"
            f"{caption_content}\n"
            "--- PATIENT'S MEDICAL REPORT (THE SUMMARY) ---\n"
            f"{report_content}"
        )
    except FileNotFoundError:
        return jsonify({"error": f"Report files for '{patient_file_base}' not found."}), 404

    messages = [
        {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is my patient report context:\n\n{report_context}\n\nPlease use this information to answer my questions."},
        {"role": "assistant", "content": "I have read your report. I am ready to help. What is your first question?"}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    
    try:
        def get_response(msgs):
            """
            Streams a clean, formatted response, buffering and removing <think> tags.
            """
            global client, api_key_queue
            
            full_response_buffer = ""
            think_tag_found = False
            
            try:
                stream = client.chat.completions.create(
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
                            parts = full_response_buffer.split("</think>", 1)
                            clean_part = parts[1].lstrip()
                            
                            if clean_part.startswith('\n'):
                                clean_part = clean_part.lstrip('\n')
                                
                            yield clean_part

            except GroqError as e:
                is_tpd_limit = e.status_code == 429 and "tpd" in str(e.message).lower()
                is_restricted = e.status_code == 400 and "organization_restricted" in str(e.message).lower()

                if is_tpd_limit or is_restricted:
                    if is_tpd_limit:
                        print(f"Key ...{api_key_queue[0][-4:]} exhausted (TPD). Rotating key.")
                    if is_restricted:
                        print(f"Key ...{api_key_queue[0][-4:]} has been BANNED (Organization Restricted). Rotating key.")
                    
                    api_key_queue.rotate(-1)
                    
                    is_fully_exhausted = True
                    for i in range(len(api_key_queue)):
                        if os.getenv(f"GROQ_API_KEY_{i+1}") != api_key_queue[i]:
                            is_fully_exhausted = False
                            break
                    
                    if is_fully_exhausted:
                        yield "ERROR: All API keys have hit their daily token limits or have been restricted. Please try again tomorrow or contact support."
                        return

                    client = Groq(api_key=api_key_queue[0])
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
    print("Flask server starting...")
    print(f"Loaded {len(GROQ_API_KEYS)} API keys.")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)