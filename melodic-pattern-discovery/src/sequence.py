import numpy as np

def contains_silence(seq, thresh=0.05):
    """If more than <thresh> of <seq> is 0, return True"""
    return sum(seq==0)/len(seq) > thresh


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def is_stable(seq, max_var=10):
    mu = np.nanmean(seq)
    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0


def reduce_stability_mask(stable_mask, min_stability_length_secs, timestep):
    min_stability_length = int(min_stability_length_secs/timestep)
    num_one = 0
    indices = []
    for i,s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            if num_one < min_stability_length:
                for ix in indices:
                    stable_mask[ix] = 0
            num_one = 0
            indices = []
    return stable_mask


def get_silence_stability_mask(raw_pitch, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):
    stab_hop = int(stability_hop_secs/timestep)
    reverse_raw_pitch = np.flip(raw_pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(raw_pitch))]
    stable_mask_2 = [is_stable(reverse_raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(reverse_raw_pitch))]
    
    silence_mask = raw_pitch == 0

    stable_zipped = zip(stable_mask_1, np.flip(stable_mask_2))
    
    stable_mask = np.array([int(any([x,y])) for x,y in stable_zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)

    mask = np.array([int(any([x,y])) for x,y in zip(stable_mask, silence_mask)])

    return mask


def start_with_silence(seq):
    return any([seq[0] == 0, all(seq[:100]==0)])


def min_gap(seq, length=86):
    seq2 = np.trim_zeros(seq)
    m1 = np.r_[False, seq2==0, False]
    idx = np.flatnonzero(m1[:-1] != m1[1:])
    if len(idx) > 0:
        out = (idx[1::2]-idx[::2])
        if any(o >= length for o in out):
            return True
    return False


def too_much_mask(seq, thresh=0.05):
    return sum(seq==1)/len(seq) > thresh