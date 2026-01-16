import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import config

def _normalize_color(color):
    """
    Checks if a color is in 0-255 range and normalizes it to 0-1
    for Matplotlib.
    """
    if isinstance(color, str):
        return color
        
    try:
        if any(c > 1 for c in color):
            return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        else:
            return color 
    except Exception:
        return (1, 0, 1) # Default to magenta

def _get_color(class_name):
    if class_name in config.FDI_CLASSES:
        return _normalize_color(config.FDI_COLOR)
    return _normalize_color(config.DISEASE_COLORS.get(class_name, (1, 1, 1))) # Default to white

def _draw_bbox(ax, bbox, label, score, color, bbox_format='xyxy', text_y_offset=0):
    """
    Draws a single bounding box on the axes.
    Includes a vertical offset for the text label.
    """
    if bbox_format == 'xywh':
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    else: # xyxy
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
    text_y_position = y1 - 5 - text_y_offset
    
    ax.text(
        x1, text_y_position, text,
        color='white',
        fontsize=8,
        bbox=dict(facecolor=plot_color, alpha=0.6, pad=0.5)
    )
    return x1, y2

def _draw_annotations(ax, image, annotations, is_gt=False):
    """Draws all annotations for one image."""
    ax.imshow(image)
    ax.axis('off')

    if is_gt:
        for ann in annotations:
            if ann['category_id'] >= len(config.CLASSES):
                 continue
            class_name = config.CLASSES[ann['category_id']]
            color = _get_color(class_name)
            _draw_bbox(ax, ann['bbox'], class_name, 1.0, color, bbox_format='xywh')
    
    else:
        for tooth in annotations['teeth']:
            color = _get_color(tooth['class_name'])
            bx, by = _draw_bbox(ax, tooth['bbox'], tooth['class_name'], tooth['score'], color)
            
            if tooth['recommendation']:
                rec_color = _normalize_color('red') 
                ax.text(
                    bx, by + 15,
                    f"Rec: {tooth['recommendation']}",
                    color='white',
                    fontsize=9,
                    bbox=dict(facecolor=rec_color, alpha=0.7, pad=0.5)
                )

            condition_label_offset = 0
            vertical_spacing = 12
            
            # Draw conditions for that tooth
            for cond in tooth['conditions']:
                if not cond['bbox']: continue # Skip rules
                color = _get_color(cond['class_name'])
                
                _draw_bbox(
                    ax, 
                    cond['bbox'], 
                    cond['class_name'], 
                    cond['score'], 
                    color,
                    text_y_offset=condition_label_offset
                )
                
                condition_label_offset += vertical_spacing
        
        # Draw unassigned diseases (no offset)
        for disease in annotations['unassigned']:
            color = _get_color(disease['class_name'])
            _draw_bbox(ax, disease['bbox'], disease['class_name'], disease['score'], color)


def create_comparison_image(image_pil, gt_annotations, final_predictions, output_path):
    """
    Creates and saves the 2-column comparison image.
    """
    image_np = np.array(image_pil)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.subplots_adjust(wspace=0.05)

    # Left Column: Ground Truth
    ax1.set_title("Ground Truth", fontsize=16, color='white')
    _draw_annotations(ax1, image_np, gt_annotations, is_gt=True)

    # Right Column: Ensemble Prediction
    ax2.set_title("Ensemble Prediction", fontsize=16, color='white')
    _draw_annotations(ax2, image_np, final_predictions, is_gt=False)
    
    fig.patch.set_facecolor('black')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
