import cv2

# Method 1: Simple bounding box
def draw_bbox_simple(image, xmin, ymin, xmax, ymax, color=(0, 255, 0), thickness=2):
    """
    Draw a simple bounding box on image
    
    Args:
        image: Image array
        xmin, ymin, xmax, ymax: Bounding box coordinates
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels
    
    Returns:
        Image with bounding box drawn
    """
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image


# Method 2: Bounding box with label
def draw_bbox_with_label(image, xmin, ymin, xmax, ymax, label, 
                         color=(0, 255, 0), thickness=2, 
                         font_scale=0.5, text_thickness=1):
    """
    Draw bounding box with text label
    
    Args:
        image: Image array
        xmin, ymin, xmax, ymax: Bounding box coordinates
        label: Text label to display
        color: BGR color tuple
        thickness: Box line thickness
        font_scale: Font size scale
        text_thickness: Text line thickness
    
    Returns:
        Image with labeled bounding box
    """
    # Draw rectangle
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    
    # Draw filled rectangle for text background
    cv2.rectangle(
        image,
        (xmin, ymin - text_height - baseline - 5),
        (xmin + text_width + 5, ymin),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        image,
        label,
        (xmin + 2, ymin - 5),
        font,
        font_scale,
        (255, 255, 255),  # White text
        text_thickness,
        cv2.LINE_AA
    )
    
    return image


# Method 3: Draw from YOLO format
def draw_bbox_from_yolo(image, class_id, x_center, y_center, width, height,
                       class_names=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box from YOLO format (normalized coordinates)
    
    Args:
        image: Image array
        class_id: Class ID
        x_center, y_center, width, height: YOLO normalized coordinates (0-1)
        class_names: List of class names (optional)
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Image with bounding box drawn
    """
    img_height, img_width = image.shape[:2]
    
    # Convert YOLO format to pixel coordinates
    x_center_px = int(x_center * img_width)
    y_center_px = int(y_center * img_height)
    width_px = int(width * img_width)
    height_px = int(height * img_height)
    
    # Calculate corner coordinates
    xmin = int(x_center_px - width_px / 2)
    ymin = int(y_center_px - height_px / 2)
    xmax = int(x_center_px + width_px / 2)
    ymax = int(y_center_px + height_px / 2)
    
    # Get label
    if class_names and class_id < len(class_names):
        label = class_names[class_id]
    else:
        label = f"Class {class_id}"
    
    # Draw with label
    draw_bbox_with_label(image, xmin, ymin, xmax, ymax, label, color, thickness)
    
    return image

# Method 4: Draw multiple bounding boxes with different colors
def draw_multiple_bboxes(image, bboxes, class_names=None, color_map=None):
    """
    Draw multiple bounding boxes on image
    
    Args:
        image: Image array
        bboxes: List of bounding boxes, each as dict with keys:
                {'xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence' (optional)}
        class_names: List of class names
        color_map: Dict mapping class_id to color, or None for random colors
    
    Returns:
        Image with all bounding boxes drawn
    """
    # Default color map
    if color_map is None:
        # Generate random colors for each class
        np.random.seed(42)
        max_class = max([bbox.get('class_id', 0) for bbox in bboxes], default=0)
        color_map = {
            i: tuple(map(int, np.random.randint(0, 255, 3)))
            for i in range(max_class + 1)
        }
    
    for bbox in bboxes:
        x_center = bbox['x_center']
        y_center = bbox['y_center']
        width = bbox['width']
        height = bbox['height']
        class_id = bbox.get('label', 0)
        
        # Get color for this class
        color = color_map.get(class_id, (0, 255, 0))
        
        # Create label
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        

        
        # Draw
        draw_bbox_from_yolo(image, label, x_center, y_center, width, height)
    
    return image