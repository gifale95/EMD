# EEG Moments Dataset (EMD)

This is the accompanying GitHub repository of the [EEG Moments Dataset (EMD)](paper_link)!!!!!!!!!!. EMD consists of 128-channel EEG responses and eye-tracking recordings of 6 human participants viewing 1,102 short naturalistic videos (3-second long; with audio track) while maintaining central fixation. EMD is the companion dataset of the [BOLD Moments Dataset (BMD)](https://doi.org/10.1038/s41467-024-50310-3), which consists of fMRI recordings of 10 human participants for the same 1,102 videos.

<p align="center">
  <img src="figure_1.jpg" alt="Figure 1" width="750"><br>
  <em><strong>Figure 1.</strong> EMD’s stimuli, stimulus metadata, experimental design, and data acquisition. (<strong>a</strong>) EMD’s stimuli consist of 1,102 short naturalistic videos. (<strong>b</strong>) Each video stimulus has rich metadata. (<strong>c</strong>) EMD consists of 6 participants with 8 data collection sessions each. During these 8 sessions, each of the 1,000 training videos were repeated 6 times, and each of the 102 testing videos were repeated 24 times. (<strong>d</strong>) Each experimental trial started with 1 s of blank screen, followed by 3 s of video presentation, 0.25 s of blank screen, and ended with 1.5 s where participants were instructed to blink. Videos were presented at the center of the screen subtending with a central red fixation cross. Participants were instructed to focus on the fixation cross. (<strong>e</strong>) We recorded neural responses throughout the video viewing experiment using 128-channel EEG. (<strong>f</strong>) We recorded monocular eye-tracking data throughout the video viewing experiment, including x gaze, y gaze, and pupil size.</em>
</p>



## 🧰 Download EMD

EMD is freely available on OpenNeuro: [https://openneuro.org/datasets/ds008257](https://openneuro.org/datasets/ds008257).


## 🚀 EMD data tutorial

Through [this interactive Colab tutorial](https://colab.research.google.com/drive/1Z5MDo8yy3sucggLQ4SMETtud2E1igRE9?usp=drive_link) in Python you will familiarize with EMD’s preprocessed EEG and eye-tracking data, as well as with the stimuli and stimulus metadata.


## ♻️ Reproducibility

This repository contains code to reproduce all [paper's results](link_paper)!!!!!!!!!!!!!.



### ⚙️ Installation

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



## ❗ Issues

If you experience problems with the dataset or code please submit an issue, or get in touch with Ale Gifford (alessandro.gifford@gmail.com).



## 📜 Citation

If you use any of our data or code, please cite:

> * Gifford AT, Oyarzo P, Zonneveld AW, Sartzetaki C, Groen IIA, Cichy RM. 2026. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!. _arXiv_, DOI: [!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1](!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
> * Lahner B, Dwivedi K, Iamshchinina P, Graumann M, Lascelles A, Roig G, Gifford AT, Pan B, Jin S, Murty AR, Kay K, Oliva A, Cichy RM. 2024. Modeling short visual events through the BOLD moments video fMRI dataset and metadata. _Nature Communications_, 15(1), 6241. DOI: [https://doi.org/10.1038/s41467-024-50310-3](https://doi.org/10.1038/s41467-024-50310-3)