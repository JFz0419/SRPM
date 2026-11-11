"""Generate the Seattle dataset in BasicTS format.

Raw data (from the RIPCN release, Zenodo record 19149017) goes to
``datasets/raw_data/Seattle/``:

    flow.npy       (T, N, 3)  hourly [flow, speed, occupancy]
    adj.npy        (N, N)     adjacency matrix

``T = 8760`` (one year, hourly) and ``N = 323`` sensors. Only the flow channel is
kept by default (``target_channel = [0]``), and the time-of-day / day-of-week
features are appended, which is the same convention used by the other traffic
datasets in this repository.
"""
import json
import os

import numpy as np

from generate_adj_mx import generate_adj_seattle

# Hyperparameters
dataset_name = 'Seattle'
raw_dir = f'datasets/raw_data/{dataset_name}'
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
channel_files = ['flow.npy']  # flow.npy 里已经是 (T, N, 3) = [flow, speed, occupancy]
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
steps_per_day = 24  # Number of time steps per day (hourly data)
frequency = 1440 // steps_per_day
domain = 'traffic flow'
feature_description = [domain, 'time of day', 'day of week']
regular_settings = {
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'TRAIN_VAL_TEST_RATIO': [0.6, 0.2, 0.2],
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': 0.0
}


def load_and_preprocess_data():
    '''Load raw channel arrays, stack them and select the target channel(s).'''
    channels = []
    for file_name in channel_files:
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(file_path):
            continue
        arr = np.asarray(np.load(file_path))
        if arr.ndim == 2:  # (T, N) -> single channel
            arr = arr[..., None]
        assert arr.ndim == 3, f'{file_name}: expect (T, N) or (T, N, C), got {arr.shape}'
        channels.append(arr.astype(np.float64))
        print(f'[load] {file_name} shape={arr.shape} '
              f'min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}')
    assert channels, f'no channel file found under {raw_dir}'
    data = np.concatenate(channels, axis=-1)  # L x N x C
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data


def add_temporal_features(data):
    '''Add time of day and day of week as features to the data.'''
    l, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(l)])
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features


def save_data(data):
    '''Save the preprocessed data to a binary file.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f'Data saved to {file_path}')


def save_graph():
    '''Save the adjacency matrix to the output directory, generating it if necessary.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    graph_file_path = os.path.join(raw_dir, 'adj_mx.pkl')
    if not os.path.exists(graph_file_path):
        generate_adj_seattle()
    import pickle
    with open(graph_file_path, 'rb') as f:
        adj_mx = pickle.load(f)
    with open(output_graph_path, 'wb') as f:
        pickle.dump(adj_mx, f)
    print(f'Adjacency matrix saved to {output_graph_path}, shape={np.shape(adj_mx)}, '
          f'nnz={(np.asarray(adj_mx) > 0).sum()}')


def save_description(data):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'shape': data.shape,
        'num_time_steps': data.shape[0],
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'feature_description': feature_description,
        'has_graph': True,
        'frequency (minutes)': frequency,
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)


def main():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Add temporal features
    data_with_features = add_temporal_features(data)

    # Save processed data
    save_data(data_with_features)

    # Copy or generate and save adjacency matrix
    save_graph()

    # Save dataset description
    save_description(data_with_features)


if __name__ == '__main__':
    main()
