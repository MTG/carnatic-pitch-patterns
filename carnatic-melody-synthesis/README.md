## Analysis/Synthesis based method to generate Carnatic Music vocal melody annotations.

## Introduction
This repository hosts an analysis/synthesis framework to generate South Indian Art Music (Carnatic) vocal melody annotations, using the brand new [Saraga Dataset](https://mtg.github.io/saraga/) (compiled within the CompMusic research project from the Carnatic and Hindustani corpora).

---

## Usage

First run `python dataset_preprocessor.py` to preprocess your input dataset before the vocal synthesis. Edit the  `main` function to run the preprocessing step on for the desired case. There is a `CarnaticDatasetPreprocessor` class which is designed for the Saraga Carnatic Dataset.

Then run `python dataset_synthesis.py` to generate the synthesized vocal melody ground-truth dataset. The `main` function may be also adapted to accomodate your dataset. In this case, some additional modifications shall be done. Take into account that the synthesis step is not immediate and you need to consider some computation time.

---

## References
* **This application reuses code from:**
    * [Essentia](https://github.com/MTG/essentia) by Music Technology Group 
    * [Spleeter](https://github.com/deezer/spleeter) by Deezer
    * Adapted version of [sms-tools](https://github.com/MTG/sms-tools) by Music Technology Group 
    * [MELODIA](https://www.justinsalamon.com/melody-extraction.html) by Justin Salamon and Emilia Gómez
    * [PredominantMelodyMakam](https://github.com/sertansenturk/predominantmelodymakam) by Hasan Sercan Atli and Sertan Şentürk

* **The following papers are considered as baseline for this implementation:**

    * *Salamon, J., Bittner, R. M., Bonada, J., Bosch, J. J., Gómez, E., & Bello,
   J. P. (2017). An analysis/synthesis framework for automatic f0 annotation of 
   multitrack datasets. Proceedings of the 18th International Society for Music 
   Information Retrieval Conference, ISMIR 2017, 71–78.*
   
    * *Bogdanov, Dmitry & Wack, Nicolas & Gómez, Emilia & Gulati, Sankalp & Herrera,
   Perfecto & Mayor, Oscar & Roma, Gerard & Salamon, Justin & Zapata, Jose & Serra, 
   Xavier. (2013). ESSENTIA: An open-source library for sound and music analysis.
   Proceedings - 21st ACM International Conference on Multimedia. 10.1145/2502081.2502229.*
   
    * *Atlı, H. S., Uyar, B., Şentürk, S., Bozkurt, B., and Serra, X. (2014). Audio
   feature extraction for exploring Turkish makam music. In Proceedings of 3rd
   International Conference on Audio Technologies for Music and Media, Ankara,
   Turkey.*
  
    * *Salamon, J., & Gomez, E. (2012). Melody extraction from polyphonic music signals
   using pitch contour characteristics. IEEE Transactions on Audio, Speech and 
   Language Processing, 20(6), 1759–1770.*
  
* **The audios used are taken from the** [**Saraga Dataset**](https://mtg.github.io/saraga/).