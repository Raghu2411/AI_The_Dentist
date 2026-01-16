import json
import math
import re
import config_master as config
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle
from reportlab.lib.pagesizes import portrait, A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black
from PIL import Image as PILImage
from groq import Groq, GroqError
import os

def get_groq_report_response(client: Groq, caption_text: str, model_id: str) -> str:
    """
    Sends a NON-STREAMING request to Groq to generate the full medical report.
    Returns the clean report text.
    """
    try:
        messages = [
            {"role": "system", "content": config.REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": caption_text}
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.7,
            max_tokens=2048,
            stream=False 
        )
        
        raw_text = chat_completion.choices[0].message.content
        return clean_llm_output(raw_text) 
            
    except GroqError as e:
        # Handle API errors
        print(f"Error during report generation: {e}")
        return f"ERROR: API call failed. {e}"
    except Exception as e:
        print(f"Error during report generation: {e}")
        return f"ERROR: An unexpected error occurred. {e}"
    
# Caption Generation

def _get_bbox_center(bbox: list) -> tuple:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def _find_closest_tooth(teeth_list: list, finding: dict) -> str:
    if not teeth_list: return "unknown"
    finding_center = _get_bbox_center(finding.get("bbox", [0,0,0,0]))
    min_distance = float('inf')
    closest_tooth_id = "unknown"
    for tooth in teeth_list:
        tooth_center = _get_bbox_center(tooth.get("bbox", [0,0,0,0]))
        distance = math.sqrt((finding_center[0] - tooth_center[0])**2 + (finding_center[1] - tooth_center[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_tooth_id = tooth.get("class_name", "unknown")
    return closest_tooth_id

def _format_section(title: str, items: list) -> str:
    if not items: return f"{title} (total: 0):\n[\n]\n"
    section_title = f"{title} (total: {len(items)}):"
    formatted_items = []
    for item in items:
        item_str = json.dumps(item, ensure_ascii=False).replace('"', "'")
        formatted_items.append(f" {item_str}")
    items_str = ",\n".join(formatted_items)
    return f"{section_title}\n[\n{items_str}\n]"

def _get_teeth_visibility(teeth_list: list) -> str:
    visibility = []
    for tooth in teeth_list:
        bbox = tooth.get("bbox", [0,0,0,0])
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        visibility.append({
            "point_2d": [round(x_center), round(y_center)],
            "tooth_id": tooth.get("class_name", "unknown"),
            "score": round(tooth.get("score", 0), 2)
        })
    return _format_section("Teeth visibility with center points", visibility)

def _get_wisdom_teeth(teeth_list: list) -> str:
    wisdom_teeth = []
    for tooth in teeth_list:
        if tooth.get("class_name") in config.WISDOM_TEETH:
            is_impacted = any(cond.get("class_name") == 'impacted' for cond in tooth.get("conditions", []))
            wisdom_teeth.append({
                "box_2d": [round(coord) for coord in tooth.get("bbox", [])],
                "tooth_id": tooth.get("class_name"),
                "is_impacted": is_impacted,
                "score": round(tooth.get("score", 0), 2)
            })
    return _format_section("Wisdom teeth detection", wisdom_teeth)

def _get_findings(teeth_list: list, unassigned_list: list, finding_type: str) -> list:
    class_set = config.PATHOLOGICAL_CLASSES if finding_type == 'pathological' else config.TREATMENT_CLASSES
    findings = []
    for tooth in teeth_list:
        tooth_id = tooth.get("class_name", "unknown")
        for cond in tooth.get("conditions", []):
            if cond.get("class_name") in class_set:
                findings.append({
                    "box_2d": [round(c) for c in cond.get("bbox", [])],
                    "tooth_id": tooth_id,
                    "label": cond.get("class_name"),
                    "score": round(cond.get("score", 0), 2)
                })
    for item in unassigned_list:
        if item.get("class_name") in class_set:
            closest_tooth_id = _find_closest_tooth(teeth_list, item)
            finding_data = {
                "box_2d": [round(c) for c in item.get("bbox", [])],
                "tooth_id": "unknown",
                "label": item.get("class_name"),
                "score": round(item.get("score", 0), 2)
            }
            if closest_tooth_id != "unknown":
                finding_data["near_tooth"] = closest_tooth_id
            findings.append(finding_data)
    return findings

def create_grounding_caption(structured_data: dict) -> str:
    """Creates the full text caption from the final prediction data."""
    teeth_list = structured_data.get("teeth", [])
    unassigned_list = structured_data.get("unassigned", [])
    caption_parts = ["This localization caption provides multi-dimensional spatial analysis..."]
    caption_parts.append(_get_teeth_visibility(teeth_list))
    caption_parts.append(_get_wisdom_teeth(teeth_list))
    pathological = _get_findings(teeth_list, unassigned_list, 'pathological')
    caption_parts.append(_format_section("Dental Pathological Findings", pathological))
    historical = _get_findings(teeth_list, unassigned_list, 'historical')
    caption_parts.append(_format_section("Historical Treatments", historical))
    return "\n\n".join(caption_parts)


#  Medical PDF Generation & LLM Output Cleaning

def clean_llm_output(raw_text: str) -> str:
    """Strips the <think>...</think> block from the start of an LLM response."""
    match = re.search(r'</think>(.*)', raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def convert_markdown_to_flowables(md_text: str, styles: dict) -> list:
    """Converts a Markdown text string into a list of ReportLab Flowables."""
    flowables = []
    lines = md_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("### "):
            flowables.append(Paragraph(line.replace("### ", "").strip(), styles['Heading3']))
        elif line.startswith("## "):
            flowables.append(Paragraph(line.replace("## ", "").strip(), styles['Heading2']))
        elif line.startswith("# "):
            flowables.append(Paragraph(line.replace("# ", "").strip(), styles['Heading1']))
        elif line in ["---", "***", "___"]:
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=black, spaceAfter=12))
        elif line.startswith("* "):
            text = "&bull; " + re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line[2:].strip())
            flowables.append(Paragraph(text, styles['Bullet']))
        elif re.match(r'^\d+\.\s+', line):
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            indented_list_style = ParagraphStyle(
                name='IndentedList',
                parent=styles['BodyText'],
                leftIndent=0.5*cm
            )
            flowables.append(Paragraph(line, indented_list_style))
        elif not line:
            flowables.append(Spacer(1, 0.25 * cm))
        else:
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            flowables.append(Paragraph(line, styles['BodyText']))
    return flowables

def create_report_pdf(original_image_path: str, predicted_image_path: str, report_text: str, pdf_save_path: str):
    """
    Creates a PDF report with side-by-side images (Actual, Predicted)
    and the formatted medical report below.
    """
    try:
        page_width, page_height = portrait(A4)
        doc = SimpleDocTemplate(pdf_save_path, pagesize=portrait(A4),
                                leftMargin=1.5*cm, rightMargin=1.5*cm,
                                topMargin=1.5*cm, bottomMargin=1.5*cm)
        story = []
        
        styles = getSampleStyleSheet()
        
        # Modify the styles
        styles['BodyText'].fontSize = 9
        styles['BodyText'].leading = 11

        styles['Heading1'].fontSize = 14
        styles['Heading1'].leading = 16
        styles['Heading1'].spaceAfter = 6
        
        styles['Heading2'].fontSize = 12
        styles['Heading2'].leading = 14
        styles['Heading2'].spaceAfter = 6
        styles['Heading2'].alignment = 1 # 1 = TA_CENTER

        styles['Heading3'].fontSize = 10
        styles['Heading3'].leading = 12
        styles['Heading3'].spaceAfter = 4
        styles['Heading3'].fontName = 'Helvetica-Bold'
        
        # Default 'Bullet' style with modification
        styles['Bullet'].parent = styles['BodyText']
        styles['Bullet'].leftIndent = 1 * cm
        styles['Bullet'].firstLineIndent = -0.5 * cm
        
        # Standard new style
        styles.add(ParagraphStyle(
            name='CaptionText', 
            parent=styles['Normal'], 
            fontSize=8, 
            leading=10
        ))

        # Prepare Images
        col_width = (page_width - 4*cm) / 2.0 # Page width minus margins, divided by 2
        
        def get_image(path, width):
            if not os.path.exists(path):
                return Paragraph(f"Image not found:\n{os.path.basename(path)}", styles['BodyText'])
            with PILImage.open(path) as pil_img:
                img_w, img_h = pil_img.size
            aspect = img_h / float(img_w)
            return Image(path, width=width, height=width * aspect)

        img_actual = get_image(original_image_path, col_width)
        img_predicted = get_image(predicted_image_path, col_width)

        # Create Image Table
        img_table_data = [
            [Paragraph("Actual (Uploaded Image)", styles['Heading2']), Paragraph("Predicted (Ensemble Result)", styles['Heading2'])],
            [img_actual, img_predicted]
        ]
        img_table = Table(img_table_data, colWidths=[col_width, col_width])
        img_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        
        story.append(img_table)
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=black, spaceAfter=1, spaceBefore=1))
        story.append(Spacer(1, 0.5 * cm))

        # Add Medical Report
        story.append(Paragraph("Medical Report", styles['Heading1']))
        report_flowables = convert_markdown_to_flowables(report_text, styles)
        story.extend(report_flowables)

        doc.build(story)
        return True
    except Exception as e:
        print(f"\n  [Error] Failed to create PDF for {os.path.basename(pdf_save_path)}: {e}")
        return False