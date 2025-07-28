# Voeg nieuw bestand toe
touch train_model.py
open train_model.py  # (of gebruik een editor zoals VS Code)

# Plak deze inhoud:
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Voorbeelddata genereren
df = pd.DataFrame({
    'Werknemer_ID': range(1, 101),
    'Leeftijd': np.random.randint(20, 65, size=100),
    'Geslacht': np.random.choice(['Man', 'Vrouw'], size=100),
    'Afdeling': np.random.choice(['HR', 'IT', 'Finance', 'Logistiek', 'Zorg'], size=100),
    'Dienstjaren': np.random.randint(0, 40, size=100),
    'Verzuimdagen_12mnd': np.random.randint(0, 30, size=100),
    'ZiekteverzuimScore': np.random.uniform(0, 1, size=100),
    'MentaleBelastingScore': np.random.uniform(0, 1, size=100),
    'FysiekeBelastingScore': np.random.uniform(0, 1, size=100),
    'WerktevredenheidScore': np.random.uniform(0, 1, size=100),
    'VerzuimKomtWaarschijnlijk': np.random.choice([0, 1], size=100, p=[0.7, 0.3])
})

# Preprocessing
X = pd.get_dummies(df[['Geslacht', 'Afdeling']]).join(df[[
    'Leeftijd', 'Dienstjaren', 'Verzuimdagen_12mnd',
    'ZiekteverzuimScore', 'MentaleBelastingScore',
    'FysiekeBelastingScore', 'WerktevredenheidScore']])
y = df['VerzuimKomtWaarschijnlijk']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ model.pkl is succesvol opgeslagen.")
