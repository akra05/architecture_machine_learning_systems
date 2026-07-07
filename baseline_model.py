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

def local_contrast_features(arr, block_size=8):
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

def process_img_stats(df):

    real_stats = []
    ai_stats = []

    for img, label in zip(df['image'], df['label']):
        arr = np.array(Image.open(io.BytesIO(img)))
        
        #stats = process_rgb_values(arr)
        stats = local_contrast_features(arr)
        
        if label == 0:
            real_stats.append(stats)
        else:
            ai_stats.append(stats)
    
    df_real = pd.DataFrame(real_stats)
    df_ai = pd.DataFrame(ai_stats)

    print(df_real.mean())
    print(df_ai.mean())

    return real_stats, ai_stats




def main():
    df = pd.read_parquet("artifacts/train_cleaned.parquet")
    analysis = process_img_stats(df)
    #print(analysis)

if __name__ == "__main__":
    main()

