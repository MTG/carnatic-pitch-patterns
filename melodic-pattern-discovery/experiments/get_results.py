import pandas as pd

from src.io import load_pkl

df = pd.DataFrame(columns=['pitch_track', 'precision', 'recall', 'f1', 'thresh'])

for experiment in ['melodia', 'melodia_mix', 'melodia_spleeter', 'melodia_spleeter_mix', 'ftanet', 'ftanet_mix', 'ftanet_western', 'ftanet_western_mix']:
	p_path = os.path.join(f'data/3_7/{experiment}/results','precisions.pkl')
	r_path = os.path.join(f'data/3_7/{experiment}/results','recall.pkl')
	t_path = os.path.join(f'data/3_7/{experiment}/results','thresh.pkl')
	f_path = os.path.join(f'data/3_7/{experiment}/results','f1.pkl')

	precisions = load_pkl(p_path)
	recalls = load_pkl(r_path)
	thresh = load_pkl(t_path)
	f1 = load_pkl(f_path)

	i = f1.index(max(f1))

	df = df.append({
		'pitch_track': experiment,
		'precision': precisions[i],
		'recall': recalls[i],
		'f1': f1[i],
		'thresh': thresh[i],
	}, ignore_index=True)

df.sort_values(by='f1', ascending=False, inplace=True)
df