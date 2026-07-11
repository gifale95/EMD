# !!!!!!!!!!!!!!!!!!!!!! PAPER TITLE !!!!!!!!!!!!!!!!!!!!!!!

Here we provide the code to reproduce all results from the paper:</br>
"[!!!!!!!!!!!!!!!!!!!!!! PAPER TITLE !!!!!!!!!!!!!!!!!!!!!!!](!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1)".</br>
Alessandro T. Gifford, Pablo Oyarzo, Anne W. Zonneveld, Christina Sartzetaki, Iris I.A. Groen, Radoslaw M. Cichy

![Figure 1](figure_1.jpg)



## 📜 Paper abstract !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Vision neuroscience has experienced a surge in the collection and use of large-scale datasets of brain responses to naturalistic images. However, static images lack the temporal dimension essential for understanding how vision is solved in the brain during dynamic real life settings. To facilitate the study of the neural correlates of dynamic visual event perception, we introduce the EEG Moments Dataset (EMD). EMD consists of 128-channel EEG responses and eye-tracking recordings of 6 human participants viewing 1,102 short naturalistic videos (3-second long) while maintaining central fixation. We show that EMD’s EEG responses well encode stimulus-related information, exhibit a temporal correspondence with the video stimuli, and have a rich representational content revealed by brain encoding models based on different feature spaces. Furthermore, complemented by the BOLD Moments Dataset (BMD) – a large-scale dataset of human functional magnetic resonance imaging (fMRI) responses for the same videos – EMD enables a spatio-temporally resolved investigation of brain responses to dynamic visual events. We release EMD’s EEG and eye-tracking data in both raw and preprocessed format, the 1,102 video stimuli, and rich stimulus metadata. Finally, we provide a code tutorial to familiarize with EMD’s preprocessed data, stimuli, and stimulus metadata.



## ♻️ Reproducibility

### 🧰 Data

The EEG Moments Dataset (EMD) is freely available on OpenNeuro: [https://openneuro.org/datasets/ds008057](https://openneuro.org/datasets/ds008057).



### ⚙️ Installation

This repository contains code to reproduce all paper's results.

To run the code, you first need to install the libraries in the [requirements.txt](https://github.com/gifale95/EMD/blob/main/requirements.txt) file within an Anaconda environment. Here, we guide you through the installation steps.

First, create an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment with the correct Python version:

```shell
conda create -n emd_env python=3.9
```

Next, download the [requirements.txt](https://github.com/gifale95/EMD/blob/main/requirements.txt) file, navigate with your terminal to the download directory, and activate the Anaconda environment previously created with:

```shell
source activate emd_env
```

Now you can install the libraries with:

```shell
pip install -r requirements.txt
```



### 📦 Code description

* **[`00a_experimental_paradigm`](https://github.com/gifale95/EMD/tree/main/00a_experimental_paradigm):** Code used to run EMD's data collection experiment.
* **[`00b_dataset_to_bids`](https://github.com/gifale95/EMD/tree/main/00b_dataset_to_bids):** Code used to convert the EEG and eye-tracking data to raw BIDS format.
* **[`00c_preprocessing`](https://github.com/gifale95/EMD/tree/main/00c_preprocessing):** Code used to preprocess the raw EEG and eye-tracking data.
* **[`paper_figure_2`](https://github.com/gifale95/EMD/tree/main/paper_figure_2):** Data quality checks of eye-tracking gaze positions.
* **[`paper_figure_3`](https://github.com/gifale95/EMD/tree/main/paper_figure_3):** Data quality checks of EEG responses, including event-related potential (ERP), noise ceiling signal to noise ratio (NCSNR), and pairwise decoding analyses.
* **[`paper_figure_4`](https://github.com/gifale95/EMD/tree/main/paper_figure_4):** Temporal correspondence between stimulus videos and EEG responses using cross-temporal representational similarity analysis (CT-RSA).
* **[`paper_figure_5`](https://github.com/gifale95/EMD/tree/main/paper_figure_5):** Encoding models of EEG responses to videos using vision- and language- based stimulus features.
* **[`paper_figure_6`](https://github.com/gifale95/EMD/tree/main/paper_figure_6):** Encoding-based fusion between EMD's EEG responses and BMD's fMRI responses.



## 🚀 EMD data tutorial

Through [this interactive Colab tutorial](https://colab.research.google.com/drive/1Z5MDo8yy3sucggLQ4SMETtud2E1igRE9?usp=drive_link) in Python you will familiarize with EMD’s preprocessed EEG and eye-tracking data, as well as with the stimuli and stimulus metadata.



## ❗ Issues

If you experience problems with the dataset or code please submit an issue, or get in touch with Ale Gifford (alessandro.gifford@gmail.com).



## 📜 Citation

If you use any of our data or code, please cite:

> * Gifford AT, Oyarzo P, Zonneveld AW, Sartzetaki C, Groen IIA, Cichy RM. 2026. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!. _arXiv_, DOI: [!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1](!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
> * Lahner B, Dwivedi K, Iamshchinina P, Graumann M, Lascelles A, Roig G, Gifford AT, Pan B, Jin S, Murty AR, Kay K, Oliva A, Cichy RM. 2024. Modeling short visual events through the BOLD moments video fMRI dataset and metadata. _Nature Communications_, 15(1), 6241. DOI: [https://doi.org/10.1038/s41467-024-50310-3](https://doi.org/10.1038/s41467-024-50310-3)