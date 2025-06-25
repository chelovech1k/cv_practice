import os
import numpy as np
from PIL import Image
import pickle
import random
from tqdm import tqdm

# Папки с изображениями и аннотациями
IMAGES_DIR = 'images_squared'
LABELS_DIR = 'labels'

# Получаем список всех изображений
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
image_files.sort()  # Для воспроизводимости и соответствия порядку

def save_txt(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def save_rgb_txt(filename, data):
    with open(filename, 'w') as f:
        for arr in data:
            flat = np.array(arr).flatten()
            f.write(' '.join(map(str, flat)))
            f.write('\n')

X = []
y = []

for img_name in tqdm(image_files, desc='Processing images'):
    img_path = os.path.join(IMAGES_DIR, img_name)
    label_name = img_name.replace('.png', '.txt')
    label_path = os.path.join(LABELS_DIR, label_name)
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        X.append(arr)
    except MemoryError:
        print(f"MemoryError: {img_name} пропущено!")
        continue

    # Определяем класс: 1 - есть люди, 0 - нет
    label = 0
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and line[0] in '01':
                    label = 1
                    break
    y.append(label)

# Перемешиваем и делим на train/test
indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

x_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
x_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

save_txt('x_train.txt', x_train)
save_txt('y_train.txt', y_train)
save_txt('x_test.txt', x_test)
save_txt('y_test.txt', y_test)

# Также можно сохранить в pickle для быстрой загрузки
save_pickle('x_train.pickle', x_train)
save_pickle('y_train.pickle', y_train)
save_pickle('x_test.pickle', x_test)
save_pickle('y_test.pickle', y_test)
save_rgb_txt('X_rgb.txt', X)

print('Готово! x_train, y_train, x_test, y_test сохранены.')
