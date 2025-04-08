# Project for H1B Lottery prediction for CS 526

## Team member
Kaixin Lu(kl461), Xinyi Peng(xp39), Dongyu Lan(dl393), Yuxin Wang(yw529), Li'an Zhu(lz244)

## Data
https://github.com/BloombergGraphics/2024-h1b-immigration-data/blob/main/README.md

## Insturction
To run this project, please do: 

1. Download the data from https://github.com/BloombergGraphics/2024-h1b-immigration-data/.
2. Unzip them in the root directory: 526-project.
3. run the code below

```
pip install -i requirements.txt
python data_targetenc.py
python logistic regression.py
python lightGBM.py
python randomForest.py
```
All the output can be seen files named *.txt in output dir.
