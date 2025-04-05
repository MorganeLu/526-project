import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def classify_region(state):
    east = {'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA',
            'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL'}
    central = {'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS',
               'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'}
    west = {'MT', 'ID', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM', 'WA', 'OR', 'CA', 'AK', 'HI'}

    if state in east:
        return 1
    elif state in central:
        return 0
    elif state in west:
        return 2
    else:  # others: PR, VI, MP, AE, XX
        return 3


class DataPreprocessor:
    def __init__(self, keep_cols=None, preprocess_cols=None, freq_encode_cols=None):
        """
        :param keep_cols: 需要考虑的列列表
        :param preprocess_cols: 需要应用自定义处理的列 (字典: {列名: 处理函数})
        :param freq_encode_cols: 需要进行频率编码的列列表
        """
        self.keep_cols = keep_cols if keep_cols else []
        self.preprocess_cols = preprocess_cols if preprocess_cols else []
        self.freq_encode_cols = freq_encode_cols if freq_encode_cols else []
        self.freq_encoding_map = {}

    def freq_encoding(self, df):
        """
        计算训练集的 Frequency Encoding 映射
        """
        df = df.copy()

        for col in self.freq_encode_cols:
            if col in df.columns:
                self.freq_encoding_map[col] = df[col].value_counts(normalize=True).to_dict()

    def transform(self, df):
        df = df.copy()
        df = df[self.keep_cols]

        df = df[
            ~((df['ben_year_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)') & 
            (df['country_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)'))
            ]

        # preprocess_cols
        for col, func in self.preprocess_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        # encoding_map
        # pre calculation
        self.freq_encoding(df)

        for col in self.freq_encode_cols:
            if col in df.columns:
                df[col] = df[col].map(self.freq_encoding_map.get(col, {})).fillna(0)

        # clarify
        df["state"] = df["state"].apply(classify_region)

        df['ben_year_of_birth'] = pd.to_numeric(df['ben_year_of_birth'], errors='coerce')
        df['lottery_year'] = pd.to_numeric(df['lottery_year'], errors='coerce')

        df['age'] = df['lottery_year'] - df['ben_year_of_birth']

        del df['lottery_year']
        del df['ben_year_of_birth']

        return df


file_path = "data/TRK_13139_FY2021.csv"
df = pd.read_csv(file_path, low_memory=False)



'''
country_of_birth: Frequency Encoding (good for imbalance)
country_of_nationality: Frequency Encoding
Age: lottery_year - ben_year_of_birth
gender: female->1, male->0
FEIN: unique for each employer, Frequency Encoding
state: west, middle, east, others
ben_multi_reg_ind: 是否多份材料同时抽签 1/0
FIRST_DECISION: 最终结果 Approved->1, other->0
'''

preprocessor = DataPreprocessor(
    keep_cols=['country_of_birth', 'country_of_nationality', 'ben_year_of_birth', 'gender',
               'FEIN', 'state', 'ben_multi_reg_ind', 'FIRST_DECISION', 'lottery_year'],
    preprocess_cols={
        'FIRST_DECISION': lambda x: 1 if str(x).lower() == 'approved' else 0,
        'gender': lambda x: 1 if str(x).lower() == 'female' else 0
    },
    freq_encode_cols=['country_of_birth', 'country_of_nationality', 'FEIN']
)

df_processed = preprocessor.transform(df)
print(df_processed.head(20))
