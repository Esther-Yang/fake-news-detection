# FAKE NEWS DETECTION USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNING

## Contents of the repository

- `liar_dataset` (to store [LIAR](https://aclanthology.org/P17-2067/ "Original LIAR dataset") and [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS "GitHub Repo of LIAR-PLUS") dataset)
  - `README.md`
  - `test.tsv`
  - `train.tsv`
  - `valid.tsv`
  - `test2.tsv`
  - `train2.tsv`
  - `valid2.tsv`

* `processed_data` (path to store processed DataFrame after running notebook 1-1 to 3-2)

  - `df_raw.pkl` (six classes)
  - `df_bi_A.pkl` (binary class with Mapping A)
  - `df_bi_B.pkl` (binary class with Mapping B)
  

* `py_files` (Python files for report writing purposes, can be ignored)

* `saved_model` (path to save the model generated in train_model notebook 4-1, 5-1, 6-1)

* `0-0` is a demo.

* The main notebooks are from `1-1` to `6-3`.

* `import_file.py` (consists of package dependencies)

* Notebooks starting with "`zz`" are miscellaneous, can be ignored.

## Description of the overall workflow

Kindly refer to [this file](https://github.com/Esther-Yang/fake-news-detection/blob/main/fake-news-detection-description.pdf 'fake-news-detection-description.pdf') for written description on collection of datasets, exploratory analysis, text preprocessing, feature extraction, selection of machine learning algorithms and method of model evaluation. The specifications of implementation are included in the aforementioned file too.
