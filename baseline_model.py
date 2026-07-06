import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def process_img_stats(df):

    real_stats = []
    ai_stats = []

    for img, label in zip(df['image'], df['label']):
        arr = np.array(Image.open(io.BytesIO(img)))
        
        stats = {
            'mean_r': float(arr[:,:,0].mean()),
            'mean_g': float(arr[:,:,1].mean()),
            'mean_b': float(arr[:,:,2].mean()),
            'std_r': float(arr[:,:,0].std()),
            'std_g': float(arr[:,:,1].std()),
            'std_b': float(arr[:,:,2].std()),
        }
        
        if label == 0:
            real_stats.append(stats)
        else:
            ai_stats.append(stats)
    
    df_real = pd.DataFrame(real_stats)
    df_ai = pd.DataFrame(ai_stats)

    print(df_real.mean())
    print(df_ai.mean())

    return real_stats, ai_stats

def is_grayscale(arr, threshold=10):
    # wenn R, G, B überall sehr ähnlich sind → Graustufen
    diff_rg = np.abs(arr[:,:,0].astype(int) - arr[:,:,1].astype(int)).mean()
    diff_rb = np.abs(arr[:,:,0].astype(int) - arr[:,:,2].astype(int)).mean()
    return diff_rg < threshold and diff_rb < threshold

real_gray = sum(1 for s in real_stats if s.get('is_grayscale', False))
ai_gray = sum(1 for s in ai_stats if s.get('is_grayscale', False))

print(f"Real Graustufen: {real_gray}/{len(real_stats)}")
print(f"AI Graustufen: {ai_gray}/{len(ai_stats)}")

def main():
    df = pd.read_parquet("artifacts/train_cleaned.parquet")
    analysis = process_img_stats(df)
    #print(analysis)

if __name__ == "__main__":
    main()

