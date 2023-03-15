## FTANet-melodic adaptation to the SCMS Carnatic Music dataset.

## Introduction
This folder contains the code to re-train, evaluate, and perform fast inference with the FTANet-melodic using the SCMS Dataset (named here FTANet-carnatic).

---

## Usage

Please note that you may need the SCMS to run some parts of the code, especially the training code. In the `train.py`, `evaluator.py` `fast_inference.py` files, remember to correctly set up the `'PATH-TO-SCMS'` (for the `fast_inference.py`, you need to set up the containing home folder instead of the direct path to the dataset).

Run `train.py` to train the FTANet-melodic using the training set of the SCMS. Three models are stored, the one with best OA (overall accuracy), another one that achieves the best RPA (raw pitch accuracy), and finally the one that obtained the lower loss.

Run `evaluator.py` to evaluate the trained model on the testing subset of the SCMS. The OA model is considered as the standard one to evaluate.

Run `fast_inference.py` to extract the pitch tracks from a list of example tracks filtered from the original Saraga Carnatic dataset. Feel free to modify the list to extract the pitch from whatever recording in the dataset.

*Comming soon:* Run `generic_inference.py` to extract the pitch from any Carnatic / Hindustani music recording. 

---

## References
* **This application reuses code from:**
    * [FTANet-melodic](https://github.com/yushuai/FTANet-melodic) original FTANet-melodic repository

* **The following papers are considered as baseline for this implementation:**

    * *Yu, S., Sun, X., Yu, Y., & Li, W. (2021). Frequency-temporal attention network for singing melody extraction. In: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 251-255)*
 
