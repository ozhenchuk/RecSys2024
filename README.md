Recommender Systems @ UCU | Capstone Project

**Evaluation checkpoints:**

HW1:

- branch
- commit

The project is based on the dataset

**Repository structure:**

- `data` - dataset and any additional data needed for the project. The dataset itself is not stored in the remote repository due to its size and can be downloaded from
- `experiments` - detailed info about each carried out experiment, including data exploration. The folder names are structured as `Exp_[project part number 1-3]_[number]_[date]_[name]`
- `scripts` - any non-model scripts that are additionally developed for the project
- `src` - source code of the project's reusable code itself. A lot of the code here start out as the positive experiments results

**Setup instructions:**

- Clone the repository (can be done via)
- Install needed packages from the `requirements.txt` by executing
- Download the full dataset from the [link](https://grouplens.org/datasets/movielens/1m/) and unpack it into the `data` folder with the whole data structure (the dataset root folder should be `data/ml_1m`)
- Congratulations, you have the project set up

**Downloading Dataset**
1. Install packages
```bash
pip install -r requirements.txt
```

2. Setup ENV variables
    
    Just copy `.env.example` to `.env`

3. Run download script
```bash
python scripts/dataset.py
```
