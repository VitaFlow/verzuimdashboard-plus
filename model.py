import pandas as pd
import joblib

def load_model(model_path='model.pkl'):
    return joblib.load(model_path)

def preprocess(df):
    df = df.copy()
    df_encoded = pd.get_dummies(df[['Geslacht', 'Afdeling']])
    numerical = df[['Leeftijd', 'Dienstjaren', 'Verzuimdagen_12mnd',
                    'ZiekteverzuimScore', 'MentaleBelastingScore',
                    'FysiekeBelastingScore', 'WerktevredenheidScore']]
    return pd.concat([df_encoded, numerical], axis=1)

def predict(model, df):
    X = preprocess(df)
    probs = model.predict_proba(X)[:, 1]
    df['Risicoscore'] = probs
    df['Risicoklasse'] = pd.cut(probs, bins=[0, 0.33, 0.66, 1], labels=['Laag', 'Midden', 'Hoog'])
    return df
