import cv2
import os
import shutil
import random


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

    frame_ann = output_dir+"/labels/"+scene+"_"+video_base+"_"+frame+".txt"
    frame_png = output_dir+"/images/"+scene+"_"+video_base+"_"+frame+".jpg"

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


def split_images_into_train_val_test(source_dir, train_ratio, val_ratio, test_ratio):
    """
    Randomly splits images from a source directory into train, val, and test directories.

    Args:
        source_dir (str): Path to the directory containing all images.
        train_ratio (float): Proportion of images for the training set (e.g., 0.7).
        val_ratio (float): Proportion of images for the validation set (e.g., 0.15).
        test_ratio (float): Proportion of images for the test set (e.g., 0.15).
    """

    # Ensure ratios sum to 1
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(image_files)

    total_images = len(image_files)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * val_ratio)

    train_files = image_files[:train_split]
    val_files = image_files[train_split:train_split + val_split]
    test_files = image_files[train_split + val_split:]

    # Create destination directories if they don't exist
    for sub_dir in ['train', 'val', 'test']:
        os.makedirs(os.path.join(source_dir, sub_dir), exist_ok=True)

    # Move files
    print(f"Moving {len(train_files)} images to 'train'...")
    for f in train_files:
        print(f'f: {f}')
        label_f = f.replace(".jpg", ".txt")
        label_dir = source_dir.replace("/images/","/labels/")
        shutil.move(os.path.join(source_dir, f), os.path.join(source_dir, 'train', f))
        shutil.move(os.path.join(label_dir, label_f), os.path.join(label_dir, 'train', label_f))

    print(f"Moving {len(val_files)} images to 'val'...")
    for f in val_files:
        print(f'f: {f}')
        label_f = f.replace(".jpg", ".txt")
        label_dir = source_dir.replace("/images/","/labels/")
        shutil.move(os.path.join(source_dir, f), os.path.join(source_dir, 'val', f))
        shutil.move(os.path.join(label_dir, label_f), os.path.join(label_dir, 'val', label_f))

    print(f"Moving {len(test_files)} images to 'test'...")
    for f in test_files:
        print(f'f: {f}')
        label_f = f.replace(".jpg", ".txt")
        label_dir = source_dir.replace("/images/","/labels/")
        shutil.move(os.path.join(source_dir, f), os.path.join(source_dir, 'test', f))
        shutil.move(os.path.join(label_dir, label_f), os.path.join(label_dir, 'test', label_f))

    print("Image splitting complete.")