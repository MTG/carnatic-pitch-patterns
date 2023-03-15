
## Repeated Motif Discovery 

This repository uses code adapted from the publication...

```Nuttall, T., Plaja, G., Pearson, L., and Serra, X. (November, 2021). The Matrix profile for motif discovery in audio-an example application in Carnatic music. Paper presented at the 15th International Symposium on Computer Music Multidisciplinary Research (CMMR), Tokyo, Japan.```

The original repository for which can be found [here](https://github.com/thomasgnuttall/carnatic-motifs-cmmr-2021).

Additional code corresponding to the experiments outlined in the TISMIR submission -  "Improving the discovery of melodic patterns in Carnatic Music with a tailored vocal pitch extraction methodology" is included here.

### Installation
Requires essentia library

Requires python <= 3.8

Install using 

`pip install -e .` 

### Data Objects
#### Audio
The two audios used for melodic pattern discovery is the vocal stem and mix recording of Koti Janmani - Akkarai Sisters, available as part of the [Saraga Dataset](https://mtg.github.io/saraga/) and downloadable using the [MIRDATA API](https://github.com/mir-dataset-loaders/mirdata).  

For some experiments, the Spleeter source separation library is applied to each of these, the code used for this can be found in `experiments/extract_pitch_melodia.py`

**It is important to update the `audio_path` variable in each configuration file (Table 1) such that it corresponds to the file downloaded on your machine.**

#### Pitch Tracks
8 pitch tracks are extracted and detailed in Table 1

|Alias|Location|Method|Configuration File|
|--|--|--|--|
|FTA-C, VOCAL|`data/ftanet/47_Koti_Janmani.csv`|FTA-NET trained on IAM applied to vocal stem|`conf/four_experiments/ftanet.yaml`|
|FTA-C, MIX|`data/ftanet_mix/47_Koti_Janmani.csv`|FTA-NET trained on IAM applied to mix|`conf/four_experiments/ftanet_mix.yaml`|
|FTA-W, VOCAL|`data/ftanet_western/47_Koti_Janmani.csv`|FTA-NET trained on western music applied to vocal stem|`conf/four_experiments/ftanet_western.yaml`|
|FTA-W, MIX|`data/ftanet_western_mix/47_Koti_Janmani.csv`|FTA-NET trained on western music applied to mix|`conf/four_experiments/ftanet_western_mix.yaml`|
|MELODIA, VOCAL|`data/melodia/47_Koti_Janmani.csv`|Melodia applied to vocal stem|`conf/four_experiments/melodia.yaml`|
|MELODIA, MIX|`data/melodia_mix/47_Koti_Janmani.csv`|Melodia applied to mix|`conf/four_experiments/melodia_mix.yaml`|
|MELODIA-S, VOCAL|`data/melodia_spleeter/47_Koti_Janmani.csv`|Melodia applied to vocal stem after Spleeter source separation |`conf/four_experiments/melodia_spleeter.yaml`|
|MELODIA-S, MIX|`data/melodia_spleeter_mix/47_Koti_Janmani.csv`|Melodia applied to mix after Spleeter source separation |`conf/four_experiments/melodia_spleeter_mix.yaml`|

**Table 1** - Pitch tracks extracted from Koti Janmani - Akkarai Sisters

#### Annotations
Expert annotations for Koti Janmani are provided by Carnatic Music vocalist, Brindha Manickavasakan and are available in `data/annotations.txt`. A loader is available at `src/io.py::load_annotations_brindha()`

### Melodic Pattern Recognition
**It is important to update the `audio_path` variable in each configuration file (Table 1) such that it corresponds to the file downloaded on your machine.**

A gridsearch is ran to optimize, ϕ, the minimum distance to the parent within returned motif groups (more information available in original paper, Nuttall et al.). This gridsearch is ran for each of the 8 pitch tracks presented in Table 1 and the output stored in the corresponding directory for each (.pkl files corresponding f1, precision, recall and ϕ for each iteration and a plot of classification metrics). To run this gridsearch:

`python experiments/gridsearch.py`

The optimum parameters for each experiment are found in the experiment configuration files in `conf/pipeline/four_experiments`. To run the pattern finding approach using these parameters:

`sh experiment.sh`

### Results
#### Returned Motifs
Final results are stored in the corresponding directory for each pitch track in a folder called `results/`. Each motif group should include a pitch plot corresponding to the returned pattern and the extracted audio (although in this repository we have included the pitch plots only).

The results discussed in the paper can be found in `data/FINAL_RESULTS/`

#### Annotations
`annotations_joined.csv` corresponds to the original annotations joined with the motif group and occurrence number of the match (if matched) for each pitch track.

### Visualisation
The plots included in the paper can be generated and using the code in `experiments/plots_for_paper.py` (this is not the cleanest of code!)

The pitch plots corresponding to each expert annotation using FTA-C, MIX and MELODIA-S, MIX can be found in `paper_plots/all_annotations/`. They are organised into folders of...

**both_match** - both FTA-C, MIX and MELODIA-S matched this pattern

**melodia_match** - only MELODIA-S matched this pattern

**ftanet_match** - only FTA-C, MIX matched this pattern

**no_match** - neither of the two processes matched this pattern


