import pandas as pd
import torch
def data_cleaning(file_path):
    '''
    read the csv file
    load track_id, frame and label then bounding box coordinates

    '''
    input_df = pd.read_csv(file_path,sep = '\s+',header=None)
 
    input_df.columns = ["track_id", "x_min", "y_min", "x_max", "y_max", "frame_id",
                  "lost","occluded","generated","label"]
    #print(input_df.shape)
    input_df = input_df[(input_df["lost"] !=1) & (input_df["occluded"] !=1)]
    #print(input_df.shape)
    input_df = input_df[(input_df["label"] =="Pedestrian")]

    #print(input_df.head(10))
    return input_df
    # print(input_df.columns)

def generate_frame_to_cordinate_map(clean_df):
    agent_position_dict = {}
    clean_df["x"] = clean_df["x_min"]+clean_df["x_max"]/2.0
    clean_df["y"] = clean_df["y_min"]+clean_df["y_max"]/2.0
    agent_position_dict = clean_df.groupby('frame_id').apply(
    lambda g: list(zip(g['track_id'], g['x'], g['y']))
    ).to_dict()
    frame_ids = sorted(agent_position_dict.keys())
    return agent_position_dict, frame_ids
    

def agent_hidden_idx_lookup(clean_df):
    agent_ids = clean_df['track_id'].unique()
    sorted_agents = sorted(agent_ids)
    look_up = {agent_id: idx for idx, agent_id in enumerate(sorted_agents)}
    return look_up

def generate_frame_sequences(frames, obs_len = 8, pred_len = 12 , stride = 1):
    sequence_length = obs_len + pred_len
    sequences =  []
    for start_idx in range(0, len(frames)-sequence_length+1 ,stride):
        window = frames[start_idx:start_idx+sequence_length]
        sequences.append(window)
    return sequences

def generate_tensor_from_frames(window,agent_pos_dicts, hidden_state_idx_lookup):
    max_no_agents = len(hidden_state_idx_lookup)
    seq_length = len(window)
    no_coordinates = 2
    input_tensor = torch.zeros(seq_length, max_no_agents,no_coordinates)
    mask_tensor = torch.zeros(seq_length,max_no_agents)
    for time_step,frame_id in enumerate(window):
        agents_in_frame = agent_pos_dicts.get(frame_id,[])
        for agent,x,y in agents_in_frame:
            hidden_state_idx = hidden_state_idx_lookup[agent]
            input_tensor[time_step,hidden_state_idx,0] = x
            input_tensor[time_step,hidden_state_idx,1] = y
            mask_tensor[time_step,hidden_state_idx] = 1

    return input_tensor, mask_tensor

def compute_social_grids(input_tensor,mask_tensor, grid_size=4, neighborhood_size =32):
    seq_length, max_no_agents , _ = input_tensor.shape
    cell_size = neighborhood_size/grid_size
    grid_tensor = torch.zeros(seq_length,max_no_agents,grid_size*grid_size,max_no_agents)
    # grid tensor is for recording interaction between multiple agents
    for time_step in range(seq_length):
        '''
        get index of all agents present in the time frame
            for each agent: (assumed to be the mathematical center of the grid)
                get the x and y coordinates
                for every other agent:
                    get coordinates
                    find the distance between them 
                    
        '''
        positions = input_tensor[time_step]     
        mask = mask_tensor[time_step]       
        active_agents = mask.nonzero(as_tuple=True)[0]
        for i in active_agents:
            xi, yi = positions[i]
            for j in active_agents:
                if i == j:
                    continue
                xj, yj = positions[j]
                dx = xj - xi
                dy = yj - yi
                half = neighborhood_size / 2
                if abs(dx) > half or abs(dy) > half:
                    continue
                col = int((dx + half)//cell_size)
                row = int((dy + half)//cell_size)
                col = min(max(col, 0), grid_size-1)
                row = min(max(row, 0), grid_size-1)
                cell_idx = row*grid_size + col
                grid_tensor[time_step, i, cell_idx, j] = 1.0
    return grid_tensor

