import os, glob
import pickle, json
import mirdata
from pathlib import Path

import config


def split_into_chunks(track, length):
    # Split audio stream into chunks of certain length
    split_track = [
        track[i * length:(i + 1) * length]
        for i in range((len(track) + length - 1) // length)
    ]
    
    return split_track


def get_multitrack_ids(track_ids, data, concerts_to_ignore):
    # Get list of multitrack audios from Saraga Carnatic dataset
    multitrack_list = []
    for track_id in track_ids:
        if data[track_id].audio_vocal_path is not None:
            if not any(concert in data[track_id].audio_vocal_path for concert in concerts_to_ignore):
                multitrack_list.append(track_id)
            else:
                print('Ignored track: ', track_id)
    
    return multitrack_list


def get_dataset_mapping(data_path):
    """
    Map the IAM melody dataset into a python dictionary
    data_path: path to the dataset foldeer
    returns dataset_mapping (dict): mapping of the songs and ids in the dataset
    """
    file_list = glob.glob(os.path.join(data_path, 'audio', '*.wav'))
    dataset_mapping = {}
    for track in file_list:
        filename = track.split('/')[-1]
        song_name = '_'.join(filename.split('_')[:-1])
        track_id = filename.split('_')[-1].replace('.wav', '')
        # id_list = [x.split('/')[-1].replace('.wav', '').split('_')[-1] for x in file_list if song_name in x]
        if dataset_mapping.get(song_name) is None:
            dataset_mapping[song_name] = [track_id]
        else:
            dataset_mapping[song_name].append(track_id)
    
    return dataset_mapping


def get_artist_song_mapping(dataset_path, output_path):
    artist_song_mapping = {}
    saraga_carnatic = mirdata.initialize('saraga_carnatic', data_home=dataset_path)
    track_ids = saraga_carnatic.track_ids
    saraga_data = saraga_carnatic.load_tracks()
    multitrack_list = get_multitrack_ids(
        track_ids,
        saraga_data,
        concerts_to_ignore=['Akkarai', 'Sundar', 'V Shankarnarayanan']
    )
    
    for track in multitrack_list:
        artist_name = saraga_data[track].metadata.get('album_artists')[0].get('name')
        formatted_id = '_'.join(track.split('_')[1:])
        if artist_song_mapping.get(artist_name) is None:
            artist_song_mapping[artist_name] = [formatted_id]
        else:
            artist_song_mapping[artist_name].append(formatted_id)
    
    dataset_mapping = get_dataset_mapping(
        data_path=os.path.join(Path().absolute(), 'resources', 'Saraga-Carnatic-Melody-Synth')
    )
    
    artists_tracks_mapping = {}
    for artist in artist_song_mapping.keys():
        for song in artist_song_mapping[artist]:
            if artists_tracks_mapping.get(artist) is None:
                print(artist)
                artists_tracks_mapping[artist] = [song + '_' + x for x in dataset_mapping[song]]
            else:
                artists_tracks_mapping[artist] = artists_tracks_mapping[artist] + [song + '_' + x for x in
                                                                                   dataset_mapping[song]]

    # print(artists_tracks_mapping['Vidya Subramanian'])
    
    with open(os.path.join(output_path, 'artists_to_track_mapping.pkl'), 'wb') as map_file:
        pickle.dump(artists_tracks_mapping, map_file)
    
    with open(os.path.join(output_path, 'artists_to_track_mapping.json'), 'w') as map_file:
        json.dump(artists_tracks_mapping, map_file)


if __name__ == '__main__':
    get_artist_song_mapping(
        dataset_path=config.DATASET_PATH,
        output_path=
        )
