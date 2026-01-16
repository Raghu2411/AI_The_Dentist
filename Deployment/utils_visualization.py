import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import config_master as config

def _normalize_color(color):
    """Normalizes color to 0-1 range for Matplotlib."""
    if isinstance(color, str):
        return color
    try:
        if any(c > 1 for c in color):
            return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        return color 
    except Exception:
        return (1, 0, 1) # Default magenta

def _get_color(class_name):
    """Gets the correct color for a class name."""
    if class_name in config.FDI_CLASSES:
        return _normalize_color((0, 1, 0)) # Green
    
    # Simple color mapping for diseases
    disease_colors = {
        'calculus': (1, 0.5, 0),    # Orange
        'caries': (1, 0, 0),        # Red
        'crown': (0, 1, 1),         # Cyan
        'impacted': (0.8, 0.6, 1),  # Lavender
        'implant': (0, 0, 1),       # Blue
        'periapical radiolucency': (1, 1, 0), # Yellow
        'rc-treated': (0.5, 0, 0.5),  # Purple
        'restoration': (1.0, 0.843, 0.0), # gold
        'root-stump': (1, 0.8, 0.8)   # Pink
    }
    return _normalize_color(disease_colors.get(class_name, (1, 1, 1)))

def _draw_bbox(ax, bbox, label, score, color, y_offset=0):
    """
    Draws a single bounding box on the axes.
    Includes a y_offset (in lines) to stack text labels vertically.
    """
    x1, y1, x2, y2 = bbox
    plot_color = _normalize_color(color)
        
    rect = patches.Rectangle(
        (x1, y1), (x2 - x1), (y2 - y1),
        linewidth=1.5,
        edgecolor=plot_color,
        facecolor='none'
    )
    ax.add_patch(rect)
    
    text = f"{label} ({score:.2f})"
    text_y_position = y1 - 5 - (y_offset * 15)  # Stack predictings vertically
    
    ax.text(
        x1, text_y_position, text,
        color='white',
        fontsize=8,
        bbox=dict(facecolor=plot_color, alpha=0.6, pad=0.5)
    )

def draw_predictions_on_image(image_path: str, structured_predictions: dict, output_path: str):
    """
    Loads an image, draws all structured predictions on it, and saves to output_path.
    """
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    
    fig, ax = plt.subplots(1, figsize=(24, 12))
    ax.imshow(image_np)
    ax.axis('off')
    
    # Draw tooth boxes
    for tooth in structured_predictions['teeth']:
        label_offset = 0
        
        # Draw the main tooth box and its label
        color = _get_color(tooth['class_name'])
        _draw_bbox(ax, tooth['bbox'], tooth['class_name'], tooth['score'], color, y_offset=label_offset)
        
        label_offset += 1
        
        # Draw conditions for that tooth + stacking labels
        for cond in tooth['conditions']:
            if not cond['bbox']: continue
            color = _get_color(cond['class_name'])
            _draw_bbox(ax, cond['bbox'], cond['class_name'], cond['score'], color, y_offset=label_offset)
            label_offset += 1
    
    # Draw unassigned diseases
    for disease in structured_predictions['unassigned']:
        color = _get_color(disease['class_name'])
        _draw_bbox(ax, disease['bbox'], disease['class_name'], disease['score'], color, y_offset=0)

    fig.patch.set_facecolor('black')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)