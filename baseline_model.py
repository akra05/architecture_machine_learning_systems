import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def process_rgb_values(arr):
    return {
            'mean_r': float(arr[:,:,0].mean()),
            'mean_g': float(arr[:,:,1].mean()),
            'mean_b': float(arr[:,:,2].mean()),
            'std_r': float(arr[:,:,0].std()),
            'std_g': float(arr[:,:,1].std()),
            'std_b': float(arr[:,:,2].std()),
        }

def local_contrast_stats(arr, block_size=8):
    gray = arr.mean(axis=2)  # RGB → Graustufen
    h, w = gray.shape
    
    variances = []
    means = []
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            variances.append(block.var())
            means.append(block.mean())
    
    variances = np.array(variances)
    means = np.array(means)
    
    return [
        variances.mean(),          # durchschnittlicher Kontrast
        variances.std(),           # wie ungleichmäßig ist der Kontrast
        variances.max(),           # stärkster lokaler Kontrast
        variances.min(),           # schwächster lokaler Kontrast
        np.percentile(variances, 25),
        np.percentile(variances, 75),
        means.std(),               # wie ungleichmäßig ist die Helligkeit
    ]

def saturation_stats(arr):
    saturation_list = []
    for row in arr:
        for rgb in row:
            max_val = max(rgb[0],rgb[1],rgb[2])
            
            if max_val == 0:
                continue
            else:
                saturation_stats = (max_val - min(rgb[0],rgb[1],rgb[2]))/max_val
                saturation_list.append(saturation_stats)

    saturation_list = np.array(saturation_list)

    return saturation_list.mean()

def fft_features(arr):
    gray = arr.mean(axis=2)  # RGB → Graustufen
    
    fft = np.fft.fft2(gray)
    magnitude = np.abs(np.fft.fftshift(fft))
    log_magnitude = np.log1p(magnitude)
    
    h, w = log_magnitude.shape
    cy, cx = h//2, w//2
    
    # radiale Durchschnitte - wie viel Energie bei welcher Frequenz
    features = []
    for r in range(1, 33):  # 32 Frequenzbänder
        y, x = np.ogrid[:h, :w]
        mask = (np.sqrt((x-cx)**2 + (y-cy)**2).astype(int) == r)
        features.append(float(log_magnitude[mask].mean()))
    
    return features  # 32 Features

def process_img_stats(df):

    X = []
    y = []
    counter = 0

    for img, label in zip(df['image'], df['label']):
        if counter % 100 == 0:
            print("Bild: "+f"{counter}"+" from "+f"{df.shape[0]}"+" is processed")
        arr = np.array(Image.open(io.BytesIO(img)))
        
        features = []
        features += list(process_rgb_values(arr).values())  # 6 Features
        features += local_contrast_stats(arr)               # 7 Features
        features.append(float(saturation_stats(arr)))       # 1 Feature
        features += fft_features(arr)                       # 32 Features
        
        X.append(features)  # eine Zeile = ein Bild = 46 Features
        y.append(int(label))
        counter += 1

    return X, y

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(X,y):

    # Normalisieren - wichtig weil Features verschiedene Skalen haben
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_scaled, y)

    # Speichern
    os.makedirs('artifacts/baseline', exist_ok=True)
    joblib.dump(model, 'artifacts/baseline/model.pkl')
    joblib.dump(scaler, 'artifacts/baseline/scaler.pkl')
    print(model)

from sklearn.metrics import classification_report

def validate_model(model,scaler):
    

    # Validation Daten laden und Features extrahieren
    df_val = pd.read_parquet('data/validation/')
    df_val['label'] = df_val['source_class'].apply(lambda x: 0 if x == 0 else 1)

    X_val = []
    y_val = []
    for img, label in zip(df_val['image'], df_val['label']):
        arr = np.array(Image.open(io.BytesIO(img)))
        features = []
        features += list(process_rgb_values(arr).values())
        features += local_contrast_stats(arr)
        features.append(float(saturation_stats(arr)))
        features += fft_features(arr)
        X_val.append(features)
        y_val.append(int(label))

    X_val_scaled = scaler.transform(X_val)  # transform, nicht fit_transform!

    y_pred = model.predict(X_val_scaled)
    print(classification_report(y_val, y_pred))

    # FPR berechnen
    real_mask = np.array(y_val) == 0
    fpr = (y_pred[real_mask] == 1).mean()
    print(f"False Positive Rate: {fpr:.3f}")

def main():
    
    df = pd.read_parquet("artifacts/train_cleaned.parquet")
    X, y = process_img_stats(df)
    train_model(X,y)
    print(pd.Series(y).value_counts())
    model = joblib.load('artifacts/baseline/model.pkl')
    scaler = joblib.load('artifacts/baseline/scaler.pkl')
    validate_model(model,scaler)
    

if __name__ == "__main__":
    main()

