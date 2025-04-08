import pandas as pd


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
    def __init__(self, keep_cols=None, preprocess_cols=None, freq_encode_cols=None, target_encode_cols=None, smooth_alpha=5):
        """
        :param keep_cols: features we need to use
        :param preprocess_cols: columns easy to transform or label
        :param freq_encode_cols: columns with frequency encoding
        :param target_encode_cols: columns with target encoding
        """
        self.keep_cols = keep_cols if keep_cols else []
        self.preprocess_cols = preprocess_cols if preprocess_cols else []
        self.freq_encode_cols = freq_encode_cols if freq_encode_cols else []
        self.target_encode_cols = target_encode_cols if target_encode_cols else []
        self.smooth_alpha = smooth_alpha

        self.freq_encoding_map = {}
        self.target_encoding_map = {}
        self.global_target_mean = {}

    def fit(self, df):
        """ Calculate the Frequency Encoding and Target Encoding mapping of train data """
        df = df.copy()
        self.freq_encoding_map = {}
        self.target_encoding_map = {}
        self.global_target_mean = {}

        # preprocess_cols
        for col, func in self.preprocess_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        df = df[
            ~(
                (df['ben_year_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)')
                &
                (df['country_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)')
            )
        ]

        # frequency
        for col in self.freq_encode_cols:
            if col in df.columns:
                self.freq_encoding_map[col] = df[col].value_counts(normalize=True).to_dict()

        # target
        for col in self.target_encode_cols:
            if col in df.columns:
                global_mean = df['FIRST_DECISION'].mean()
                self.global_target_mean[col] = global_mean

                encoding_map = df.groupby(col)['FIRST_DECISION'].mean()
                counts = df.groupby(col)['FIRST_DECISION'].count()

                # smoothing
                smoothed_encoding = (counts * encoding_map + self.smooth_alpha * global_mean) / (counts + self.smooth_alpha)
                self.target_encoding_map[col] = smoothed_encoding.to_dict()

    def transform(self, df):
        df = df.copy()
        df = df[self.keep_cols]

        df = df[
            ~(
                (df['ben_year_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)')
                &
                (df['country_of_birth'] == '(b)(3) (b)(6) (b)(7)(c)')
            )
        ]

        for col, func in self.preprocess_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        df['is_foreign_national'] = (df['country_of_birth'] != df['country_of_nationality']).astype(int)
        del df['country_of_nationality']

        # encoding_map
        for col in self.freq_encode_cols:
            if col in df.columns:
                df[col] = df[col].map(self.freq_encoding_map.get(col, {})).fillna(0)

        for col in self.target_encode_cols:
            if col in df.columns:
                df[col] = df[col].map(self.target_encoding_map.get(col, {}))
                df[col] = df[col].fillna(self.global_target_mean.get(col, 0.5))  # mean to fill the new fein

        # state - label mapping
        df["state"] = df["state"].apply(classify_region)

        # age (need to be standarlized for logistics)
        df['ben_year_of_birth'] = pd.to_numeric(df['ben_year_of_birth'], errors='coerce')
        df['lottery_year'] = pd.to_numeric(df['lottery_year'], errors='coerce')

        df['age'] = df['lottery_year'] - df['ben_year_of_birth']

        del df['lottery_year']
        del df['ben_year_of_birth']

        return df


if __name__ == "__main__":
    # train data
    file_path_2021 = "data/TRK_13139_FY2021.csv"
    file_path_2022 = "data/TRK_13139_FY2022.csv"

    df_2021 = pd.read_csv(file_path_2021, low_memory=False)
    df_2022 = pd.read_csv(file_path_2022, low_memory=False)

    # combine 2021 and 2022
    df = pd.concat([df_2021, df_2022], axis=0, ignore_index=True)

    # test data
    df_test = pd.read_csv("data/TRK_13139_FY2023.csv", low_memory=False)

    '''
    country_of_birth: Frequency Encoding (good for imbalance)
    # deleted: country_of_nationality: Frequency Encoding
    Age: lottery_year - ben_year_of_birth
    gender: female->1, male->0
    FEIN: unique for each employer, target Encoding
    state: west, middle, east, others
    ben_multi_reg_ind: whether hand in same materials for one year 1/0
    FIRST_DECISION: final result Approved->1, other->0
    is_foreign_national: whether the nationality and the birth place is the same
    '''

    '''
    Especially for LR:
    Age: needs to be standarlized
    state: one-hot needed
    '''

    preprocessor = DataPreprocessor(
        keep_cols=['country_of_birth', 'country_of_nationality', 'ben_year_of_birth', 'gender',
                'FEIN', 'state', 'ben_multi_reg_ind', 'FIRST_DECISION', 'lottery_year'],
        preprocess_cols={
            'FIRST_DECISION': lambda x: 1 if str(x).lower() == 'approved' else 0,
            'gender': lambda x: 1 if str(x).lower() == 'female' else 0
        },
        freq_encode_cols=['country_of_birth'],
        target_encode_cols=['FEIN']
    )

    preprocessor.fit(df)  # Encoding Mapping with train

    df_train_processed = preprocessor.transform(df)  # train
    df_test_processed = preprocessor.transform(df_test)  # test without new mapping

    # print(df_train_processed.head(20))
    # print(df_test_processed.head(20))
    # print((df_train_processed['is_foreign_national']==1).sum())
    # print((df_test_processed['is_foreign_national']==1).sum())

    df_train_processed.to_csv("21-22.csv")
    df_test_processed.to_csv("23.csv")
