import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class PitchProcessor(object):
    def __init__(self, hop_size=128, frame_size=2048, gap_len=25, pitch_preproc=True, voicing=False):

        self.hop_size = hop_size
        self.frame_size = frame_size
        self.gap_len = gap_len

        self.pitch_preproc = pitch_preproc  # Flag for pitch preprocessing
        self.voicing = voicing  # Flag for pitch voicing filtering on audio

        self.sample_rate = 44100  # The standard sampling frequency for Saraga audios

    def pre_processing(self, audio, extracted_pitch):
        # Load audio and adapt pitch length
        extracted_pitch = extracted_pitch[:-2]

        # Zero pad the audio so the length is multiple of 128
        if len(audio) % self.hop_size != 0:
            zero_pad = np.zeros(int((self.hop_size * np.ceil(len(audio) / self.hop_size))) - len(audio))
            audio = np.concatenate([audio, zero_pad])

        # Parsing time stamps and pitch values of the extracted pitch data
        time_stamps = [x[0] for x in extracted_pitch]
        pitch_values = [x[1] for x in extracted_pitch]
        
        # Remove values out of IAM vocal range bounds (Venkataraman et al, 2020)
        pitch_values = self.limiting(
            pitch_values=pitch_values,
            limit_up=600,
            limit_down=80,
        )

        # To enhance the automatically extracted pitch curve
        if self.pitch_preproc:
            # Interpolate gaps shorter than 250ms (Gulati et al, 2016)
            pitch_values = self.interpolate_below_length(
                arr=pitch_values,
                val=0.0,
            )
            # Smooth pitch track a bit
            pitch_values = self.smoothing(pitch_values, sigma=1)

        # To remove audio content from unvoiced areas
        if self.voicing:
            voiced_samples = []
            for sample in pitch_values:
                if sample > 0.0:
                    voiced_samples = np.concatenate([voiced_samples, self.hop_size * [1]])
                else:
                    voiced_samples = np.concatenate([voiced_samples, self.hop_size * [0]])

            # Set to 0 audio samples which are not voiced while detecting silent zone onsets
            audio_modif = audio.copy()
            silent_zone_on = 1
            silent_onsets = []
            for idx, voiced_sample in enumerate(voiced_samples):
                if voiced_sample == 0:
                    audio_modif[idx] = 0.0
                    if silent_zone_on == 0:
                        silent_onsets.append(idx)
                        silent_zone_on = 1
                else:
                    if silent_zone_on == 1:
                        silent_onsets.append(idx)
                        silent_zone_on = 0

            # Remove first onset if first sample is voiced
            if voiced_samples[0] == 1:
                silent_onsets = silent_onsets[1:] if silent_onsets[0] == 0 else silent_onsets

            # A bit of fade out at sharp gaps
            for onset in silent_onsets:
                # Make sure that we don't run out of bounds
                if onset + self.hop_size < len(audio_modif):
                    audio_modif[onset - (self.hop_size * 16):onset + (self.hop_size * 16)] = self.smoothing(
                        audio_modif[onset - (self.hop_size * 16):onset + (self.hop_size * 16)], sigma=5
                    )

            return audio_modif, pitch_values, time_stamps

        else:
            return audio, pitch_values, time_stamps

    def interpolate_below_length(self, arr, val):
        """
        Interpolate gaps of value, <val> of
        length equal to or shorter than <gap> in <arr>
        :param arr: Array to interpolate
        :type arr: np.array
        :param val: Value expected in gaps to interpolate
        :type val: number
        :return: interpolated array
        :rtype: np.array
        """
        s = np.copy(arr)
        is_zero = s == val
        cumsum = np.cumsum(is_zero).astype('float')
        diff = np.zeros_like(s)
        diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
        for i, d in enumerate(diff):
            if d <= self.gap_len:
                s[int(i-d):i] = np.nan
        interp = pd.Series(s).interpolate(method='linear', axis=0)\
                             .ffill()\
                             .bfill()\
                             .values
        return interp

    @staticmethod
    def smoothing(pitch_values, sigma=1):
        return gaussian_filter1d(pitch_values, sigma=sigma)
    
    @staticmethod
    def limiting(pitch_values, limit_up, limit_down):
        pitch_values = [x if x < limit_up else 0.0 for x in pitch_values]
        pitch_values = [x if x > limit_down else 0.0 for x in pitch_values]
        return pitch_values

    @staticmethod
    def fix_octave_errors(pitch_track):
        for i in np.arange(len(pitch_track) - 1):
            if (pitch_track[i + 1] != 0) and (pitch_track[i] != 0):
                ratio = pitch_track[i + 1] / pitch_track[i]
                octave_range = np.log10(ratio) / np.log10(2)
                if 0.95 < octave_range < 1.05:
                    pitch_track[i + 1] = pitch_track[i + 1] / 2
    
        return pitch_track
