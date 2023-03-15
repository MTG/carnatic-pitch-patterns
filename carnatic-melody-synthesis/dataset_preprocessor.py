import mirdata
import essentia.standard as estd
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

from core import DatasetPreprocessor

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

from utils import *
from config import DATASET_PATH

np.random.seed(280490)


class CarnaticDatasetPreprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path=None,
                 tanpura_dataset_path=os.path.join(Path().absolute(), 'resources', 'tanpura_tracks'),
                 chunks_path=os.path.join(Path().absolute(), 'resources', 'tmp_carnatic')):
        
        super().__init__(dataset_path, tanpura_dataset_path, chunks_path)
        self.dataset_path = dataset_path
        self.tanpura_dataset_path = tanpura_dataset_path
        self.chunks_path = chunks_path
    
    def get_mix_splits(self):
        # Create output folder if it does not exist
        if not os.path.exists(self.chunks_path):
            os.mkdir(self.chunks_path)
        
        # Initialize Saraga Carnatic dataset and get list of multitrack audio subset
        saraga_carnatic = mirdata.initialize('saraga_carnatic', data_home=self.dataset_path)
        track_ids = saraga_carnatic.track_ids
        saraga_data = saraga_carnatic.load_tracks()
        concerts_to_ignore = ['Akkarai', 'Sundar', 'Shankaranarayanan']
        multitrack_list = get_multitrack_ids(track_ids, saraga_data, concerts_to_ignore)
        
        computed_songs = []
        for track_id in tqdm(multitrack_list):
            
            # Get track to format
            track = saraga_data[track_id]
            
            # Get song name
            song_name = '_'.join(track_id.split('_')[1:])
            
            # Get artist name
            concert_name = track.metadata.get('concert')[0].get('title')
            artist_name = concert_name.replace(' at Arkay', '').replace(' in Arkay', '').replace(' ', '_')
            
            # Get tanpura audio from the synthesized tanpura dataset
            tanpura_filename = os.path.join(self.tanpura_dataset_path, 'tanpura_' + str(artist_name) + '.wav')
            audio_tanpura = estd.MonoLoader(
                filename=tanpura_filename)()
            # Get mix
            audio_mix = estd.MonoLoader(
                filename=track.audio_path)()
            
            # Get splits
            split_mix = split_into_chunks(audio_mix, len(audio_tanpura))
            
            # Account for repeated song names in the dataset
            if song_name not in computed_songs:
                computed_songs.append(song_name)
                song_name_aux = song_name
            else:
                song_name_aux = song_name + '_2'
                computed_songs.append(song_name_aux)
            
            for split_id, mix in enumerate(split_mix):
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_mix.wav'),
                    rate=44100,
                    data=np.array(mix))
    
    def split_dataset_in_chunks(self):
        # Create output folder if it does not exist
        if not os.path.exists(self.chunks_path):
            os.mkdir(self.chunks_path)
        
        # Initialize Saraga Carnatic dataset and get list of multitrack audio subset
        saraga_carnatic = mirdata.initialize('saraga_carnatic', data_home=self.dataset_path)
        track_ids = saraga_carnatic.track_ids
        saraga_data = saraga_carnatic.load_tracks()
        concerts_to_ignore = ['Akkarai', 'Sundar', 'Shankaranarayanan']
        multitrack_list = get_multitrack_ids(track_ids, saraga_data, concerts_to_ignore)
        
        computed_songs = []
        for track_id in tqdm(multitrack_list):
            
            # Get track to format
            track = saraga_data[track_id]
            
            # Get song name
            song_name = '_'.join(track_id.split('_')[1:])
            
            # Get artist name
            concert_name = track.metadata.get('concert')[0].get('title')
            artist_name = concert_name.replace(' at Arkay', '').replace(' in Arkay', '').replace(' ', '_')
            
            # Get tanpura audio from the synthesized tanpura dataset
            tanpura_filename = os.path.join(self.tanpura_dataset_path, 'tanpura_' + str(artist_name) + '.wav')
            audio_tanpura = estd.MonoLoader(
                filename=tanpura_filename)()
            # Get voice
            audio_vocal = estd.MonoLoader(
                filename=track.audio_vocal_path)()
            # Get violin
            audio_violin = estd.MonoLoader(
                filename=track.audio_violin_path)()
            # Get mridangam right
            audio_mridangam_right = estd.MonoLoader(
                filename=track.audio_mridangam_right_path)()
            # Get mridangam left
            audio_mridangam_left = estd.MonoLoader(
                filename=track.audio_mridangam_left_path)()
            
            # Get splits
            split_mridangam_left = split_into_chunks(audio_mridangam_left, len(audio_tanpura))
            split_mridangam_right = split_into_chunks(audio_mridangam_right, len(audio_tanpura))
            split_violin = split_into_chunks(audio_violin, len(audio_tanpura))
            split_vocal = split_into_chunks(audio_vocal, len(audio_tanpura))
            split_tanpura = [audio_tanpura] * len(split_vocal)
            
            if song_name not in computed_songs:
                computed_songs.append(song_name)
                song_name_aux = song_name
            else:
                song_name_aux = song_name + '_2'
                computed_songs.append(song_name_aux)
            
            for split_id, (tanpura, vocal, violin, mri_right, mri_left) in enumerate(
                    zip(split_tanpura, split_vocal, split_violin, split_mridangam_right, split_mridangam_left)):
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_tanpura.wav'),
                    rate=44100,
                    data=np.array(tanpura))
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_vocal.wav'),
                    rate=44100,
                    data=np.array(vocal))
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_violin.wav'),
                    rate=44100,
                    data=np.array(violin))
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_mridangam_right.wav'),
                    rate=44100,
                    data=np.array(mri_right))
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_mridangam_left.wav'),
                    rate=44100,
                    data=np.array(mri_left))
    
    def clean_tracks(self, separator2):
        tmp_audios = glob.glob(os.path.join(self.chunks_path, '*.wav'))
        # Iterate over tracks to clean
        for track in tqdm(tmp_audios):
            if 'vocal' in track:
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='vocals')
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=track,
                    rate=44100,
                    data=np.array(audio_clean))
            if 'mridangam' in track:
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='accompaniment')
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=track,
                    rate=44100,
                    data=np.array(audio_clean))
            if 'violin' in track:
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='accompaniment')
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=track,
                    rate=44100,
                    data=np.array(audio_clean))
                    
                    
class NoMultitrackDatasetPreprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path=None,
                 chunks_path=os.path.join(Path().absolute(), 'resources', 'tmp_hindustani')):
        
        super().__init__(dataset_path, chunks_path)
        self.dataset_path = dataset_path
        self.chunks_path = chunks_path
    
    def split_dataset_in_chunks(self):
        # Create output folder if it does not exist
        if not os.path.exists(self.chunks_path):
            os.mkdir(self.chunks_path)
        
        # Get list of tracks
        hindustani_audio_files = []
        hindustani_concerts = glob.glob(os.path.join(self.dataset_path, '*'))
        for i in hindustani_concerts:
            songs = glob.glob(os.path.join(self.dataset_path, i, '*'))
            for j in songs:
                song_filename = j + '/' + j.split('/')[-1] + '.mp3.mp3'
                hindustani_audio_files.append(song_filename)
        
        computed_songs = []
        for track in tqdm(hindustani_audio_files):
            
            # Get song name
            song_name = track.split('/')[-1].replace('.mp3.mp3', '')
            
            # Get mix
            audio_mix = estd.MonoLoader(
                filename=track)()
            
            # Get splits
            split_mix = split_into_chunks(audio_mix, 44100*30)
            
            if song_name not in computed_songs:
                song_name_aux = song_name.replace(' ', '_')
                computed_songs.append(song_name_aux)
            else:
                song_name_aux = song_name.replace(' ', '_') + '_2'
                computed_songs.append(song_name_aux)
            
            for split_id, split in enumerate(split_mix):
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_vocal.wav'),
                    rate=44100,
                    data=np.array(split))
                write(
                    filename=os.path.join(
                        self.chunks_path, song_name_aux + '_' + str(split_id) + '_accompaniment.wav'),
                    rate=44100,
                    data=np.array(split))
    
    def clean_tracks(self, separator2):
        tmp_audios = glob.glob(os.path.join(self.chunks_path, '*.wav'))
        # Iterate over tracks to clean
        for track in tqdm(tmp_audios):
            if 'vocal' in track:
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='vocals')
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=track,
                    rate=44100,
                    data=np.array(audio_clean))
            if 'accompaniment' in track:
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='accompaniment')
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=track,
                    rate=44100,
                    data=np.array(audio_clean))


# Run python3 data_preprocessor.py to pre-process the input dataset to be synthesized
if __name__ == '__main__':
    dataset_path = DATASET_PATH
    dataset_creator = CarnaticDatasetPreprocessor(
        dataset_path=dataset_path,
        tanpura_dataset_path=os.path.join(Path().absolute(), 'resources', 'tanpura_tracks'),
        chunks_path=os.path.join(Path().absolute(), 'resources', 'tmp_carnatic'))
