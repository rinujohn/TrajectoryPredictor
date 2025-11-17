from pathlib import Path
import pandas as pd

def get_mov_paths_pathlib(root_dir):
    """
    Extract all MP4 file paths using pathlib
    
    Args:
        root_dir: Root directory to search
    
    Returns:
        List of Path objects pointing to MP4 files
    """
    root = Path(root_dir)
    mp4_files = list(root.rglob("*.mov"))  # rglob = recursive glob
    return mp4_files


def get_ann_path_from_video(video_path):
    """
    Get corresponding annotation from video path
    
    Args:
        video_path: path to video.mov file
    
    Returns:
        ann_path: path to annotations.txt file associated to video_path
    """
    ann_path = str(video_path).replace("video.mov", "annotations.txt")
    ann_path = ann_path.replace("/videos/","/annotations/")
    return ann_path


def convert_txt_to_df(txt_file):
    """
    Convert annotation.txt file to dataframe

    Args:
        txt_file: path to annotations.txt file

    Return
        df: dataframe corresponding to annotations
    """
    df = pd.read_csv(txt_file, sep=r'\s+', header=None,
                 names=['Track ID', 'xmin', 'ymin', 'xmax','ymax','frame','lost','occluded', 'generated', 'label'],
                 engine='python')
    return df