import os
import re
import glob
import random
import numpy as np
import essentia.standard as estd
from tqdm import tqdm
from pathlib import Path

from core import VocalMelodySynthesis
from synthesis import Synthesizer
from pitch_track_processing import PitchProcessor

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

np.random.seed(280490)


class CarnaticMelodySynthesis(VocalMelodySynthesis):
    def __init__(self, hop_size=128, frame_size=2048,
                 data_path=os.path.join(Path().absolute(), 'resources', 'tmp_carnatic'),
                 output_dataset_path=os.path.join(Path().absolute(), 'resources', 'Saraga-Carnatic-Melody-Synth'),
                 output_home=os.path.join(Path().absolute(), 'resources', 'output')):
        
        super().__init__(hop_size, frame_size, data_path, output_dataset_path, output_home)
        self.hop_size = hop_size  # default hopSize of PredominantMelody
        self.frame_size = frame_size  # default frameSize of PredominantMelody
        
        self.data_path = data_path  # Path where clean chunks are
        self.output_dataset_path = output_dataset_path  # Path to store the saraga synth dataset
        self.output_home = output_home  # Output folder for test outputs
        
        self.sample_rate = 44100  # The standard sampling frequency for Saraga audio
    
    def get_dataset(self, pitch_preproc=True, voicing=False):
        
        if not os.path.exists(self.output_dataset_path):
            # Create dataset folder
            os.mkdir(self.output_dataset_path)
            os.mkdir(os.path.join(self.output_dataset_path, 'audio'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'melody'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'activations'))
        
        # Get vocal audio paths
        total_vocal_tracks = glob.glob(os.path.join(self.data_path, '*_vocal.wav'))
        total_ids = [x.split('/')[-1].replace('_vocal.wav', '') for x in total_vocal_tracks]
        computed_vocal_paths = glob.glob(os.path.join(self.output_dataset_path, 'audio/*.wav'))
        computed_ids = [x.split('/')[-1].replace('.wav', '') for x in computed_vocal_paths]
        
        remaining_ids = [x for x in total_ids if x not in computed_ids]
        remaining_vocal_paths = [
            self.data_path + '/' + x + '_vocal.wav' for x in remaining_ids]
                
        voice_cleaner = Separator('spleeter:2stems')
        
        # Iterate over remaining tracks to synthesize
        for track in tqdm(random.sample(remaining_vocal_paths, 1000)):
            print(track)
            _, _, _ = self.generate_synthesized_mix(
                filename=track,
                separator=voice_cleaner,
                pitch_preproc=pitch_preproc,
                voicing=voicing)
    
    def generate_synthesized_mix(self, filename, separator, pitch_preproc, voicing):
        # Get file id from filename
        file_id = filename.split('/')[-1].replace('_vocal.wav', '')
        
        # Load audio with Spleeter's AudioAdapter
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(
            filename,
            sample_rate=self.sample_rate)
        
        # Run vocal separation on vocal audio
        prediction = separator.separate(waveform)
        audio = prediction['vocals']
        
        # To mono, energy filering and apply EqualLoudness for a better pitch extraction
        audio_mono = audio.sum(axis=1) / 2
        audio_mono_filt = self.filter_audio(audio=audio_mono, coef=0.00125)  # Energy filter to remove background noise
        audio_mono_eqloud = estd.EqualLoudness(sampleRate=self.sample_rate)(audio_mono_filt)
        
        # Extract pitch using PredominantMelodyMakam algorithm
        est_time, est_freq = self.extract_pitch_pmm(audio=audio_mono_eqloud)
        pitch = [[x, y] for x, y in zip(est_time, est_freq)]
        
        # Preprocessing analyzed audio and pitch
        preprocessor = PitchProcessor(
            pitch_preproc=pitch_preproc,
            voicing=voicing,
            gap_len=0.25)
        audio, pitch_processed, time_stamps_processed = preprocessor.pre_processing(
            audio=audio_mono,
            extracted_pitch=pitch)
        
        # Get freq limits to compute minf0
        tmp_est_freq = [x for x in est_freq if x > 20]
        if len(tmp_est_freq) > 0:
            minf0 = min(tmp_est_freq) - 20
        else:
            minf0 = 0
        
        # Synthesize vocal track
        synthesizer = Synthesizer(
            model='hpr',
            minf0=minf0,
            maxf0=max(pitch_processed) + 50)
        synthesized_audio, pitch_track = synthesizer.synthesize(
            filtered_audio=audio,
            pitch_track=pitch_processed)
        
        # Get synthesized mix
        synthesized_audio_mix = self.mix(
            filename=filename,
            synthesized_voice=synthesized_audio)
        
        # Get vocal activations
        start_times, end_times = self.get_activations(time_stamps_processed, pitch_track)
        
        # Aminorate octave errors
        pitch_processed = PitchProcessor.fix_octave_errors(pitch_processed)
        
        # Get percentage of voiced time-stamps
        pitch_track_len = len(pitch_processed)
        voiced_values = []
        num_unvoiced = 0
        for i in pitch_processed:
            if i == 0.0:
                num_unvoiced += 1
            else:
                voiced_values.append(i)
        unvoiced_rate = (num_unvoiced / pitch_track_len) * 100
        if unvoiced_rate < 100:
            voiced_mean = np.mean(voiced_values)
        else:
            voiced_mean = 0.0
        # Store voiced and within a standard range of values
        if unvoiced_rate < 90:
            if voiced_mean < 475:
                # Write synthesized audio to file
                tmp_wav = 'audio/' + file_id + '.wav'
                self.save_audio_to_dataset(tmp_wav, synthesized_audio_mix)
                
                # Write csv melody annotation to file
                tmp_txt = 'annotations/melody/' + file_id + '.csv'
                self.save_pitch_track_to_dataset(tmp_txt, time_stamps_processed, pitch_track)
                
                # Write lab activations to file
                tmp_lab = 'annotations/activations/' + file_id + '.lab'
                self.save_activation_to_dataset(tmp_lab, start_times, end_times)
                
                return synthesized_audio_mix, pitch_track, time_stamps_processed
        else:
            print('UNVOICED TRACK! Skipping...')
            return [], [], []
    
    def mix(self, filename, synthesized_voice):
        # Get instrument lineup
        audio_len = len(synthesized_voice)
        violin_track = self.load_and_filter_audio(filename, 'violin', audio_len)
        mridangam_right_track = self.load_and_filter_audio(filename, 'mridangam_right', audio_len)
        mridangam_left_track = self.load_and_filter_audio(filename, 'mridangam_left', audio_len)
        tanpura_track = self.load_and_filter_audio(filename, 'tanpura', audio_len)
        stem_dict = {
            'vocals': synthesized_voice,
            'violin': violin_track,
            'mridangam_right': mridangam_right_track,
            'mridangam_left': mridangam_left_track,
            'tanpura': tanpura_track}
        
        mix_weights = self.analyze_mix_stft(
            mix=self.load_and_filter_audio(filename, 'mix', audio_len),
            stems=stem_dict)

        # Get mix
        synthesized_audio_mix = [
            x * mix_weights['vocals'] +
            y * mix_weights['violin'] +
            z * mix_weights['mridangam_right'] +
            w * mix_weights['mridangam_left'] +
            t * mix_weights['tanpura'] for x, y, z, w, t in zip(
                synthesized_voice,
                violin_track,
                mridangam_right_track,
                mridangam_left_track,
                tanpura_track)]
        return synthesized_audio_mix

    def load_and_filter_audio(self, filename_in, instrument, audio_len):
        # Get instrument lineup
        filename = filename_in.replace("vocal.wav", instrument + ".wav")
        if ('tanpura' in instrument) or ('mix' in instrument):
            audio_mono = estd.MonoLoader(filename=filename)()
            audio_mono_processed = np.array(audio_mono[:audio_len + 1], dtype='float64')

        else:
            coef = 0.00075 if 'violin' in instrument else 0.00125
            audio_mono = estd.MonoLoader(filename=filename)()
            audio_mono_processed = np.array(audio_mono[:audio_len + 1], dtype='float64')
            audio_mono_processed_filt = self.filter_audio(audio=audio_mono_processed, coef=coef)

        return audio_mono_processed_filt

class NoMulitrackMelodySynthesis(VocalMelodySynthesis):
    def __init__(self, hop_size=128, frame_size=2048,
                 data_path=os.path.join(Path().absolute(), 'resources', 'tmp_dataset_hindustani'),
                 output_dataset_path=os.path.join(Path().absolute(), 'resources', 'Hindustani-Synth-Dataset_2'),
                 mixing_weights={'vocals': 1, 'accompaniment': 1}):
        
        super().__init__(hop_size, frame_size, data_path, output_dataset_path)
        self.hop_size = hop_size  # default hopSize of PredominantMelody
        self.frame_size = frame_size  # default frameSize of PredominantMelody
        
        self.mixing_weights = mixing_weights
        
        self.data_path = data_path  # Path where clean chunks are
        self.output_dataset_path = output_dataset_path  # Path to store the saraga synth dataset
        
        self.sample_rate = 44100  # The standard sampling frequency for Saraga audio
    
    def get_dataset(self, pitch_preproc=True, voicing=False):
        
        if not os.path.exists(self.output_dataset_path):
            # Create dataset folder
            os.mkdir(self.output_dataset_path)
            os.mkdir(os.path.join(self.output_dataset_path, 'audio'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'melody'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'activations'))
        
        # Get vocal audio paths
        total_vocal_tracks = glob.glob(os.path.join(self.data_path, '*_vocal.wav'))
        total_ids = [x.split('/')[-1].replace('_vocal.wav', '') for x in total_vocal_tracks]
        computed_vocal_paths = glob.glob(os.path.join(self.output_dataset_path, 'audio/*.wav'))
        computed_ids = [x.split('/')[-1].replace('.wav', '') for x in computed_vocal_paths]
        
        remaining_ids = [x for x in total_ids if x not in computed_ids]
        remaining_vocal_paths = [
            self.data_path + '/' + x + '_vocal.wav' for x in remaining_ids
        ]
        
        print(len(remaining_vocal_paths))
        
        voice_cleaner = Separator('spleeter:2stems')
        
        # Iterate over remaining tracks to synthesize
        for track in tqdm(remaining_vocal_paths):
            _, _, _ = self.generate_synthesized_mix(
                filename=track,
                separator=voice_cleaner,
                pitch_preproc=pitch_preproc,
                voicing=voicing
            )
    
    def generate_synthesized_mix(self, filename, separator, pitch_preproc, voicing):
        # Get file id from filename
        file_id = filename.split('/')[-1].replace('_vocal.wav', '')
        
        # Load audio with Spleeter's AudioAdapter
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(
            filename,
            sample_rate=self.sample_rate
        )
        
        # Run vocal separation on vocal audio
        prediction = separator.separate(waveform)
        audio = prediction['vocals']
        
        # To mono, energy filering and apply EqualLoudness for a better pitch extraction
        audio_mono = audio.sum(axis=1) / 2
        audio_mono_filt = self.filter_audio(audio=audio_mono, coef=0.00125)  # Energy filter to remove background noise
        audio_mono_eqloud = estd.EqualLoudness(sampleRate=self.sample_rate)(audio_mono_filt)
        
        # Extract pitch using PredominantMelodyMakam algorithm
        est_time, est_freq = self.extract_pitch_pmm(audio=audio_mono_eqloud)
        pitch = [[x, y] for x, y in zip(est_time, est_freq)]
        
        # Preprocessing analyzed audio and pitch
        preprocessor = PitchProcessor(
            pitch_preproc=pitch_preproc,
            voicing=voicing,
            gap_len=25,
        )
        audio, pitch_processed, time_stamps_processed = preprocessor.pre_processing(
            audio=audio_mono,
            extracted_pitch=pitch,
        )
        
        # Get freq limits to compute minf0
        tmp_est_freq = [x for x in est_freq if x > 20]
        if len(tmp_est_freq) > 0:
            minf0 = min(tmp_est_freq) - 20
        else:
            minf0 = 0
        
        # Synthesize vocal track
        synthesizer = Synthesizer(
            model='hpr',
            minf0=minf0,
            maxf0=max(pitch_processed) + 50,
        )
        synthesized_audio, pitch_track = synthesizer.synthesize(
            filtered_audio=audio,
            pitch_track=pitch_processed,
        )
        
        # Get synthesized mix
        synthesized_audio_mix = self.mix(
            filename=filename,
            synthesized_voice=synthesized_audio
        )
        
        # Get vocal activations
        start_times, end_times = self.get_activations(time_stamps_processed, pitch_track)
        
        if len(start_times) > 2:
            # Write synthesized audio to file
            tmp_wav = 'audio/' + file_id + '.wav'
            self.save_audio_to_dataset(tmp_wav, synthesized_audio_mix)
            
            # Write csv melody annotation to file
            tmp_txt = 'annotations/melody/' + file_id + '.csv'
            self.save_pitch_track_to_dataset(tmp_txt, time_stamps_processed, pitch_track)
            
            # Write lab activations to file
            tmp_lab = 'annotations/activations/' + file_id + '.lab'
            self.save_activation_to_dataset(tmp_lab, start_times, end_times)
            
            return synthesized_audio_mix, pitch_track, time_stamps_processed
        else:
            print('UNVOICED TRACK! Skipping...')
            return [], [], []
    
    def mix(self, filename, synthesized_voice):
        # Get instrument lineup
        filename_accompaniment = filename.replace("vocal.wav", "accompaniment.wav")
        
        # Load audios and trim to synthesized voice length
        accompaniment_mono = estd.MonoLoader(filename=filename_accompaniment)()
        accompaniment_mono_processed = np.array(accompaniment_mono[:len(synthesized_voice) + 1], dtype='float64')
        
        # Get mix
        synthesized_audio_mix = [
            x * self.mixing_weights['vocals'] +
            y * self.mixing_weights['accompaniment'] for x, y in zip(
                synthesized_voice,
                accompaniment_mono_processed,)]
        
        return synthesized_audio_mix


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Run python3 dataset_synthesis.py to perform synthesis over input data
if __name__ == '__main__':
    carnatic_synthesizer = CarnaticMelodySynthesis(
        data_path=os.path.join(Path().absolute(), 'resources', 'tmp_carnatic'),
        output_dataset_path=os.path.join(Path().absolute(), 'resources', 'Saraga-Carnatic-Melody-Synth')
    )
    carnatic_synthesizer.get_dataset(
        pitch_preproc=True,
        voicing=False,
    )
