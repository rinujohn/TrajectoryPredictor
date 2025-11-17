import cv2

def bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format
    
    Args:
        xmin, ymin, xmax, ymax: Bounding box coordinates in pixels
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        x_center, y_center, width, height: Normalized coordinates (0-1)
    """
    # Calculate center coordinates
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    
    # Calculate width and height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

    
def create_yolo_files_from_stanford(output_dir, video_path, frame_cnt):
    """
    Given output dir and stanford drone dataset video path and frame, generate
    annotation file name and image file name

    Args:
        output_dir: Output directory where yolo data gets stored
        video_path: Path to video.mov file
        frame_cnt: Frame count for image

    Return:
        frame_ann: file name for frame yolo annotation
        frame_png: file name for frame image
    """

    scene = video_path.split('/')[-3]
    video_base = video_path.split('/')[-2]
    frame = f"frame_{frame_cnt}"

    frame_ann = output_dir+"/"+scene+"_"+video_base+"_"+frame+".txt"
    frame_png = output_dir+"/"+scene+"_"+video_base+"_"+frame+".jpg"

    return frame_ann, frame_png

def create_yolo_ann(output_file, yolo_data, output_img, frame):

    with open(output_file, "w") as f:
        for data in yolo_data:
            x_center = data["x_center"]
            y_center = data["y_center"]
            width = data["width"]
            height = data["height"]
            label = data["label"]
            
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    cv2.imwrite(output_img, frame)