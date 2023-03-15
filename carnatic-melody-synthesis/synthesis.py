import sys
import os
import numpy as np
from pathlib import Path
from scipy.signal import get_window

# Append sms tools model folder to sys.path
if os.path.join(Path().absolute(), 'sms_tools', 'models') not in sys.path:
    sys.path.append(os.path.join(Path().absolute(), 'sms_tools', 'models'))
from sms_tools.models import hpsModel as HPS
from sms_tools.models import hprModel as HPR

np.set_printoptions(threshold=sys.maxsize)

class Synthesizer(object):
    def __init__(self, model='hpr', window='hanning', M=1001, N=4096, Ns=512, H=128, t=-90,
                 minSineDur=0.001, nH=30, maxf0=1760, minf0=55, f0et=5.0, harmDevSlope=0.001, stocf=0.1):

        # Model to use
        self.model = model

        # Synthesis parameters
        self.window = window
        self.M = M
        self.N = N
        self.Ns = Ns
        self.H = H
        self.t = t
        self.minSineDur = minSineDur
        self.nH = nH
        self.maxf0 = maxf0
        self.minf0 = minf0
        self.f0et = f0et
        self.harmDevSlope = harmDevSlope
        self.stocf = stocf

        self.sample_rate = 44100  # The standard sampling frequency for Saraga audios

    def get_parameters(self):
        return {'window': self.window, 'M': self.M, 'N': self.N, 'Ns': self.Ns, 'H': self.H, 't': self.t,
                'minSineDur': self.minSineDur, 'nH': self.nH, 'minf0': self.minf0, 'maxf0': self.maxf0,
                'f0et': self.f0et, 'harmDevSlope': self.harmDevSlope, 'stocf': self.stocf}

    def synthesize(self, filtered_audio, pitch_track):

        # Get window for the stft
        w = get_window(self.window, self.M, fftbins=True)

        if self.model == 'hps':
            # Get harmonic content from audio using extracted pitch as reference
            hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x=filtered_audio, f0=pitch_track, fs=self.sample_rate, w=w,
                                                            N=self.N, H=self.H, t=self.t, nH=self.nH, minf0=self.minf0,
                                                            maxf0=self.maxf0, f0et=self.f0et,
                                                            harmDevSlope=self.harmDevSlope, minSineDur=self.minSineDur,
                                                            Ns=self.Ns, stocf=self.stocf)

            # Synthesize audio with generated harmonic content
            y, _, _ = HPS.hpsModelSynth(hfreq, hmag, hphase, stocEnv, self.Ns, self.H, self.sample_rate)

            return y, pitch_track

        if self.model == 'hpr':
            # Get harmonic content from audio using extracted pitch as reference
            hfreq, hmag, hphase, xr, f0_new = HPR.hprModelAnal(x=filtered_audio, f0=pitch_track, fs=self.sample_rate, w=w,
                                                       N=self.N, H=self.H, t=self.t, minSineDur=self.minSineDur,
                                                       nH=self.nH, minf0=self.minf0, maxf0=self.maxf0, f0et=self.f0et,
                                                       harmDevSlope=self.harmDevSlope)

            # Synthesize audio with generated harmonic content
            _, yh = HPR.hprModelSynth(hfreq, hmag, hphase, xr, self.Ns, self.H, self.sample_rate)
            
            return np.array(yh, dtype='float64'), f0_new



