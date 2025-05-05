import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_from_csv(filename):
    data = pd.read_csv(filename)
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 
                   'ST_Slope', 'AgeGroup', 'RestingBP_Category', 
                   'Cholesterol_Category', 'MaxHR_Category', 'Oldpeak_Category']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))  # .astype(str) dla bezpieczeństwa
        label_encoders[col] = le  # zapis enkoderów do późniejszego użycia

    # 4. Podział na cechy (X) i etykietę (y)
    X = data.drop('HeartDisease', axis=1).values
    y = data['HeartDisease'].values

    return X, y

