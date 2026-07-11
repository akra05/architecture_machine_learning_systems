import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import time

start = time.time()
#Hyperparameters
IMAGE_SIZE = 128

def center_crop(img, size=IMAGE_SIZE):
    w, h = img.size

    top = (h - size) // 2
    left = (w -size) // 2
    bottom = top + size
    right = left + size

    return img.crop((left, top, right, bottom))

def img_to_bytes(img):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()  # gibt bytes zurück



def transform_data(src):
    df = pd.read_parquet(src)

    # Duplikate entfernen
    df = df.drop_duplicates(subset=['image'])
    print(f"Nach Duplikat-Entfernung: {len(df)} Bilder")

    # Label mergen
    df['label'] = df['source_class'].apply(lambda x: 0 if x == 0 else 1)

    #Bilder einzeln öffnen
    cleaned_images = []

    for img in df['image']:

        img = Image.open(io.BytesIO(img))

        ratio = 128 / min(img.size)
        new_w = int(img.size[0] * ratio)
        new_h = int(img.size[1] * ratio)
        img = img.resize((new_w, new_h))

        cropped_img = center_crop(img)
        byte_image = img_to_bytes(cropped_img)
        cleaned_images.append(byte_image)

    df_cleaned = df.copy()
    df_cleaned['image'] = cleaned_images

    #save dataframe
    os.makedirs('artifacts',exist_ok=True)
    name = os.path.basename(src)  # 'data/train' → 'train'
    df_cleaned.to_parquet(f'artifacts/{name}_cleaned.parquet')



def create_diagramm(df):

    # Klassenverteilung visualisieren
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    df['source_class'].value_counts().sort_index().plot(kind='bar')
    plt.title('Klassenverteilung (original)')
    plt.xlabel('source_class')

    plt.subplot(1, 2, 2)
    counts = df['label'].value_counts().sort_index()
    counts.index = ['Real', 'AI']
    counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Klassenverteilung (binär)')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

    # Größenverteilung visualisieren
    sizes = [Image.open(io.BytesIO(b)).size for b in df['image']]
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20)
    plt.title('Breiten-Verteilung')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20)
    plt.title('Höhen-Verteilung')

    plt.tight_layout()
    plt.savefig('size_distribution.png')
    plt.show()

if __name__ == "__main__":
    src = 'data/train'
    transform_data(src)
    src = 'data/validation'
    transform_data(src)

end = time.time()
print(f"Dauer: {end - start:.2f} Sekunden")