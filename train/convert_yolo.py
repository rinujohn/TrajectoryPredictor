import cv2
import os
import sys
sys.path.append('../')

from utils.stanford import get_mov_paths_pathlib, get_ann_path_from_video, convert_txt_to_df
from utils.yolo import create_yolo_ann, bbox_to_yolo, create_yolo_files_from_stanford, split_images_into_train_val_test

# Extract video
def extract_frames(videos, label_map, output_dir, image_sz=2048, frame_interval=1):
    """Extract frames from video"""

    for video_path in videos:

        #Extract annotation
        ann = get_ann_path_from_video(video_path=video_path)
        ann_df = convert_txt_to_df(ann)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]

            #Pad img to keep consistent size
            bottom = image_sz-frame_h
            right = image_sz-frame_w
            frame = cv2.copyMakeBorder(frame, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Pad with black (0,0,0)

            # If the original image is already larger than the target size, you might want to crop or resize instead of pad.
            if bottom < 0 or right < 0:
                print("Image is larger than target size. Consider resizing or cropping.")

            if frame_count % frame_interval == 0:

                frame_ann = ann_df[ann_df['frame']==frame_count]
                frame_ann = frame_ann[frame_ann['lost']==0]
                frame_ann = frame_ann[frame_ann['occluded']==0]


                frame_ann_file, frame_jpg_file = create_yolo_files_from_stanford(output_dir, str(video_path), frame_count)

                yolo_data = []

                for id, row in frame_ann.iterrows():

                    xmin = row["xmin"]
                    ymin = row["ymin"]
                    xmax = row["xmax"]
                    ymax = row["ymax"]
                    img_width = image_sz
                    img_height = image_sz

                    x_center, y_center, width, height = bbox_to_yolo(xmin, ymin, xmax, ymax, img_width=img_width, img_height=img_height)

                    label = row["label"]

                    if label not in label_map:
                        raise KeyError
                    else:
                        label = label_map[label]

                    yolo_data.append({
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'label': label
                    })

                create_yolo_ann(frame_ann_file, yolo_data, frame_jpg_file, frame)

            frame_count+=1

        cap.release()



def convert_from_stanford():

    # Directory that contains annotations and video files
    data_path = "../data/"
    ann_path = data_path+"annotations/"
    video_path = data_path+"videos/"
    yolo_path = data_path+"yolo/"
    yolo_img_path = yolo_path+"images/"
    yolo_label_path = yolo_path+"labels/"

    os.makedirs(yolo_path, exist_ok=True)
    os.makedirs(yolo_img_path, exist_ok=True)
    os.makedirs(yolo_label_path, exist_ok=True)



    label_map = {'"Biker"': 0,
                '"Pedestrian"': 1,
                '"Skater"': 2,
                '"Cart"': 3,
                '"Car"': 4,
                '"Bus"': 5}

    videos = get_mov_paths_pathlib(video_path)

    extract_frames(videos=videos, label_map=label_map, output_dir=yolo_path, frame_interval=100)
    split_images_into_train_val_test(source_dir=yolo_img_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":

    convert_from_stanford()