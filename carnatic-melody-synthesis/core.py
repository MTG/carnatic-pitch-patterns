import csv
import os
import glob
import librosa
import numpy as np
import essentia.standard as estd
from scipy.io.wavfile import write
from pathlib import Path

from scipy.signal import get_window
from scipy.optimize import minimize

from PredominantMelodyMakam import PredominantMelodyMakam

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


class DatasetPreprocessor(object):
    # Class to preprocess the input dataset and prepare it for a optimized synthesis
    def __init__(self,
                 dataset_path=None,
                 tanpura_dataset_path=os.path.join(Path().absolute(), 'resources', 'tanpura_tracks'),
                 chunks_path=os.path.join(Path().absolute(), 'resources', 'tmp_dataset')):
        
        self.dataset_path = dataset_path
        self.tanpura_dataset_path = tanpura_dataset_path
        self.chunks_path = chunks_path
    
    def get_mix_splits(self):
        pass
    
    def split_dataset_in_chunks(self):
        pass
    
    def clean_tracks(self, separator2):
        pass
    
    @staticmethod
    def get_spleeter_prediction(separator, track_path, source):
        # Get Spleeter prediction taking model and source to obtain as input
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(
            track_path,
            sample_rate=44100
        )
        prediction = separator.separate(waveform)
        return prediction[source]
    
    @staticmethod
    def identify_unvoiced(annotation_path):
        print(annotation_path)
        for i in glob.glob(annotation_path + '*.csv'):
            with open(i, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                unvoiced = 0
                length = 0
                for row in reader:
                    value = float(row[1].replace(' ', ''))
                    if value == 0.0:
                        unvoiced += 1
                    length += 1
                unvoiced_rate = unvoiced / length
                
                if unvoiced_rate >= 0.50:
                    print(i)


class VocalMelodySynthesis(object):
    # Class to perform Analysis/Synthesis on the input dataset
    def __init__(self, hop_size=128, frame_size=2048,
                 data_path=os.path.join(Path().absolute(), 'resources', 'tmp_dataset'),
                 output_dataset_path=os.path.join(Path().absolute(), 'resources', 'Saraga-Carnatic-Melody-Synth'),
                 output_home=os.path.join(Path().absolute(), 'resources', 'output')):
        
        self.hop_size = hop_size  # default hopSize of PredominantMelody
        self.frame_size = frame_size  # default frameSize of PredominantMelody
        
        self.data_path = data_path  # Path where clean chunks are
        self.output_dataset_path = output_dataset_path  # Path to store the saraga synth dataset
        self.output_home = output_home  # Output folder for random outputs
        
        self.sample_rate = 44100  # The standard sampling frequency for Saraga audio
    
    def get_dataset(self, pitch_preproc=True, voicing=False):
        pass
    
    def generate_synthesized_mix(self, filename, separator, pitch_preproc, voicing):
        pass
    
    def mix(self, filename, synthesized_voice):
        pass
    
    def save_pitch_track_to_dataset(self, filename, est_time, est_freq):
        """
        Function to write txt annotation to file
        """
        pitchtrack_to_save = os.path.join(self.output_dataset_path, filename)
        with open(pitchtrack_to_save, 'w') as f:
            for i, j in zip(est_time, est_freq):
                f.write("{}, {}\n".format(i, j))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))
    
    def save_activation_to_dataset(self, filename, start_times, end_times):
        """
        Function to write lab activation annotation to file
        """
        activation_to_save = os.path.join(self.output_dataset_path, filename)
        with open(activation_to_save, 'w') as f:
            for i, j in zip(start_times, end_times):
                f.write("{}, {}, singer\n".format(i, j))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))
    
    def save_audio_to_dataset(self, filename, audio):
        """
        Function to write wav audio to file
        """
        audio_to_save = os.path.join(self.output_dataset_path, filename)
        write(audio_to_save, self.sample_rate, np.array(audio))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))
    
    def filter_audio(self, audio, coef):
        """
        Code taken from Baris Bozkurt's MIR teaching notebooks
        """
        audio_modif = audio.copy()
        start_indexes = np.arange(0, audio.size - self.frame_size, self.hop_size, dtype=int)
        num_windows = start_indexes.size
        w = get_window('blackman', self.frame_size)
        energy = []
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + self.frame_size] * w
            energy.append(np.sum(np.power(x_win, 2)))
        
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + self.frame_size] * w
            energy_frame = np.sum(np.power(x_win, 2))
            if energy_frame < np.max(energy) * coef:
                audio_modif[start_indexes[k]:start_indexes[k] + self.frame_size] = np.zeros(self.frame_size)
        
        return audio_modif
    
    def extract_pitch_melodia(self, audio):
        # Running melody extraction with MELODIA
        pitch_extractor = estd.PredominantPitchMelodia(frameSize=self.frame_size, hopSize=self.hop_size)
        est_freq, _ = pitch_extractor(audio)
        est_freq = np.append(est_freq, 0.0)
        est_time = np.linspace(0.0, len(audio) / self.sample_rate, len(est_freq))
        
        return est_time, est_freq
    
    def extract_pitch_pmm(self, audio):
        # Running melody extraction with PMM
        pmm = PredominantMelodyMakam(hop_size=self.hop_size, frame_size=self.frame_size)
        output = pmm.run(audio=audio)
        
        # Organizing the output
        pitch_annotation = output['pitch']
        est_time = [x[0] for x in pitch_annotation]
        est_freq = [x[1] for x in pitch_annotation]
        
        return est_time, est_freq
    
    def analyze_mix_stft(self, mix, stems):
        """
        Code taken from Bittner's et al. MedleyDB Python framework
        """
        
        mix_audio = self.get_feature_stft(mix)[:-10]
        
        instrument_stems = []
        line_up = list(stems.keys())
        for instrument in line_up:
            instrument_stems.append(stems[instrument])
        stem_audio = np.array(
            [self.get_feature_stft(_)[:len(mix_audio)] for _ in instrument_stems])
        
        # Apply constraints
        bounds = []
        for i in line_up:
            if 'vocals' in i:
                bounds.append((4, 5))
            if 'tanpura' in i:
                bounds.append((1, 3))
            else:
                bounds.append((1, 5))
        bounds = tuple(bounds)
        res = minimize(
            self.linear_model, x0=np.ones((len(line_up),)), args=(stem_audio.T, mix_audio.T),
            bounds=bounds)
        coefs = res['x']
        
        mixing_coeffs = {
            str(i): float(c) for i, c in zip(line_up, coefs)}
        if mixing_coeffs['violin'] > mixing_coeffs['vocals']:
            mixing_coeffs['violin'] = mixing_coeffs['vocals']
        return mixing_coeffs
    
    @staticmethod
    def get_activations(time_stamps, pitch_track):
        silent_zone_on = True
        start_times = []
        end_times = []
        for idx, value in enumerate(pitch_track):
            if value == 0:
                if not silent_zone_on:
                    end_times.append(time_stamps[idx - 1])
                    silent_zone_on = True
            else:
                if silent_zone_on:
                    start_times.append(time_stamps[idx])
                    silent_zone_on = False
        
        return start_times, end_times
    
    @staticmethod
    def get_feature_stft(audio):
        """
        Code taken from Bittner's et al. MedleyDB Python framework
        """
        nfft = 8192
        feature = np.abs(
            librosa.stft(audio, n_fft=nfft, hop_length=nfft, win_length=nfft)
        )
        return feature
    
    @staticmethod
    def get_feature_audio(filename):
        """
        Code taken from Bittner's et al. MedleyDB Python framework
        """
        sr = 8192
        y, fs = librosa.load(filename, mono=True, sr=sr)
        feature = y ** 2.0
        return feature
    
    @staticmethod
    def linear_model(x, A, y):
        """
        Code taken from Bittner's et al. MedleyDB Python framework
        """
        return np.linalg.norm(np.dot(A, x) - y, ord=2)