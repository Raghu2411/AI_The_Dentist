import json
import math
import config_and_prompts

def _get_bbox_center(bbox: list) -> tuple:
    """Calculates the center (x, y) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def _find_closest_tooth(teeth_list: list, finding: dict) -> str:
    """
    Finds the 'class_name' of the tooth closest to the unassigned finding.
    Returns "unknown" if no teeth are found.
    """
    if not teeth_list:
        return "unknown"

    finding_center = _get_bbox_center(finding.get("bbox", [0,0,0,0]))
    
    min_distance = float('inf')
    closest_tooth_id = "unknown"
    
    for tooth in teeth_list:
        tooth_center = _get_bbox_center(tooth.get("bbox", [0,0,0,0]))
        
        # Euclidean distance
        distance = math.sqrt(
            (finding_center[0] - tooth_center[0])**2 +
            (finding_center[1] - tooth_center[1])**2
        )
        
        if distance < min_distance:
            min_distance = distance
            closest_tooth_id = tooth.get("class_name", "unknown")
            
    return closest_tooth_id

def _format_section(title: str, items: list) -> str:
    """
    Helper function to format a list of findings into the
    required string format, similar to the reference script's 
    generate_section.
    """
    if not items:
        return f"{title} (total: 0):\n[\n]\n"
    
    section_title = f"{title} (total: {len(items)}):"
    
    formatted_items = []
    for item in items:
        item_str = json.dumps(item, ensure_ascii=False)
        item_str = item_str.replace('"', "'")
        formatted_items.append(f" {item_str}") # Add indentation
    
    items_str = ",\n".join(formatted_items)
    return f"{section_title}\n[\n{items_str}\n]"

def _get_teeth_visibility(teeth_list: list) -> str:
    """Generates the 'Teeth visibility' section."""
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
    """Generates the 'Wisdom teeth detection' section."""
    wisdom_teeth = []
    for tooth in teeth_list:
        if tooth.get("class_name") in config_and_prompts.WISDOM_TEETH:
            # Check if 'impacted' is in its conditions list
            is_impacted = any(
                cond.get("class_name") == 'impacted' 
                for cond in tooth.get("conditions", [])
            )
            
            wisdom_teeth.append({
                "box_2d": [round(coord) for coord in tooth.get("bbox", [])],
                "tooth_id": tooth.get("class_name"),
                "is_impacted": is_impacted,
                "score": round(tooth.get("score", 0), 2)
            })
    return _format_section("Wisdom teeth detection", wisdom_teeth)

def _get_findings(teeth_list: list, unassigned_list: list, finding_type: str) -> list:
    """
    A generic function to get Pathological Findings OR Historical Treatments
    based on the `finding_type` parameter.
    Uses the _find_closest_tooth() helper
    to assign findings from the 'unassigned_list' to the nearest tooth.
    """
    if finding_type == 'pathological':
        class_set = config_and_prompts.PATHOLOGICAL_CLASSES
    else:
        class_set = config_and_prompts.TREATMENT_CLASSES
        
    findings = []
    
    # Add all findings already assigned by IoU
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

    # Add unassigned findings, but try to find the closest tooth
    for item in unassigned_list:
        if item.get("class_name") in class_set:
            # Find the closest tooth to report its proximity
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

def create_grounding_caption(image_data: dict) -> str:
    """
    Takes one image's data from the ensemble JSON and creates
    the full text prompt for the LLM.
    """
    predictions = image_data.get("predictions", {})
    teeth_list = predictions.get("teeth", [])
    unassigned_list = predictions.get("unassigned", [])
    
    caption_parts = []
    
    # The standard prefix for Ground Report
    caption_parts.append(
        "This localization caption provides multi-dimensional spatial analysis of anatomical "
        "structures and pathological findings for this panoramic dental X-ray image, including:"
    )
    
    # Teeth Visibility
    caption_parts.append(_get_teeth_visibility(teeth_list))
    
    # Wisdom Teeth
    caption_parts.append(_get_wisdom_teeth(teeth_list))
    
    # Pathological Findings (Caries, Impacted, etc.)
    pathological_findings = _get_findings(teeth_list, unassigned_list, 'pathological')
    caption_parts.append(_format_section("Dental Pathological Findings", pathological_findings))
    
    # Historical Treatments (Crown, Implant, etc.)
    historical_treatments = _get_findings(teeth_list, unassigned_list, 'historical')
    caption_parts.append(_format_section("Historical Treatments", historical_treatments))
    
    # Join all parts with double new lines
    return "\n\n".join(caption_parts)