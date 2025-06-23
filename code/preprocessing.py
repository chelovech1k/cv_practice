import os
import glob
import cv2 as cv
from PIL import Image
from scipy.io import loadmat

# === 1. Конвертация .seq в .png ===
def save_img(dname, fn, i, frame, out_dir):
    # Проверяем, что кадр не пустой
    if frame is not None:
        cv.imwrite(f'{out_dir}/{os.path.basename(dname)}_{os.path.basename(fn).split(".")[0]}_{i}.png', frame)
    else:
        print(f'Пустой кадр: {fn}, {i}')

def convert_seq_to_png(seq_path, out_dir='images'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Если передан путь к конкретному файлу
    if seq_path.endswith('.seq'):
        fn = seq_path
        dname = os.path.dirname(fn)
        print(f'Открываю {fn}')
        cap = cv.VideoCapture(fn)
        if not cap.isOpened():
            print(f'Не удалось открыть {fn}')
            return
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_img(dname, fn, i, frame, out_dir)
            i += 1
        cap.release()
        print(f'Converted {fn}, кадров: {i}')
        return
    
    # Если передан путь к папке
    # Абсолютный путь к папке
    base_path = os.path.abspath(seq_path.replace('/**/*.seq', ''))
    found = False
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.seq'):
                found = True
                fn = os.path.join(root, file)
                dname = os.path.dirname(fn)
                print(f'Открываю {fn}')
                cap = cv.VideoCapture(fn)
                if not cap.isOpened():
                    print(f'Не удалось открыть {fn}')
                    continue
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    save_img(dname, fn, i, frame, out_dir)
                    i += 1
                cap.release()
                print(f'Converted {fn}, кадров: {i}')
    if not found:
        print(f'Не найдено .seq файлов в {base_path}')

# === 2. Преобразование .png в квадратные изображения ===
def squarify_images(img_dir='images', frame_size=(640, 640)):
    for frame in sorted(glob.glob(f'{img_dir}/*.png')):
        new_frame = Image.new('RGB', frame_size, 'white')
        new_frame.paste(Image.open(frame), (0, 0))
        new_frame_path = frame.replace('.png', '_squared.png')
        new_frame.save(new_frame_path, 'png')
        print(f'Saved squared {new_frame_path}')

# === 3. Конвертация .vbb в .txt ===
def convertBoxFormat(box, frame_size):
    (box_x_left, box_y_top, box_w, box_h) = box
    (image_w, image_h) = frame_size
    dw = 1. / image_w
    dh = 1. / image_h
    x = (box_x_left + box_w / 2.0) * dw
    y = (box_y_top + box_h / 2.0) * dh
    w = box_w * dw
    h = box_h * dh
    return (x, y, w, h)

def vbb_to_txt(ann_path, out_dir='labels', frame_size=(640, 640), classes=['person', 'people']):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Если передан путь к конкретному .vbb файлу
    if ann_path.endswith('.vbb'):
        caltech_annotation = ann_path
        
        # Определяем идентификатор набора из пути к файлу
        if 'set' in os.path.dirname(caltech_annotation):
            set_name = os.path.basename(os.path.dirname(caltech_annotation))
            set_nr = set_name.replace('set', '')
            dataset = 'train' if int(set_nr) < 6 else 'test'
            set_id = f'{dataset}{set_nr}'
        else:
            # Если не можем определить набор, используем 'unknown'
            set_id = 'unknown'
        
        print(f'Обработка аннотации: {caltech_annotation}')
        vbb = loadmat(caltech_annotation)
        obj_lists = vbb['A'][0][0][1][0]
        obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
        video_id = os.path.splitext(os.path.basename(caltech_annotation))[0]
        
        for frame_id, obj in enumerate(obj_lists):
            if len(obj) > 0:
                labels = ''
                for pedestrian_id, pedestrian_pos in zip(obj['id'][0], obj['pos'][0]):
                    pedestrian_id = int(pedestrian_id[0][0]) - 1
                    pedestrian_pos = pedestrian_pos[0].tolist()
                    if obj_lbl[pedestrian_id] in classes and 30 < pedestrian_pos[3] <= 80:
                        class_index = classes.index(obj_lbl[pedestrian_id])
                        yolo_box_format = convertBoxFormat(pedestrian_pos, frame_size)
                        labels += f'{class_index} ' + ' '.join([str(n) for n in yolo_box_format]) + '\n'
                if not labels:
                    continue
                image_id = f'{set_id}_{video_id}_{frame_id}_squared'
                label_file = open(f'{out_dir}/{image_id}.txt', 'w')
                label_file.write(labels)
                label_file.close()
                print(f'Annotation for {image_id} saved')
        
        return
    
    # Если передан путь к директории
    for caltech_set in sorted(glob.glob(f'{ann_path}/set*')):
        set_nr = os.path.basename(caltech_set).replace('set', '')
        dataset = 'train' if int(set_nr) < 6 else 'test'
        set_id = f'{dataset}{set_nr}'
        for caltech_annotation in sorted(glob.glob(f'{caltech_set}/*.vbb')):
            vbb = loadmat(caltech_annotation)
            obj_lists = vbb['A'][0][0][1][0]
            obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
            video_id = os.path.splitext(os.path.basename(caltech_annotation))[0]
            for frame_id, obj in enumerate(obj_lists):
                if len(obj) > 0:
                    labels = ''
                    for pedestrian_id, pedestrian_pos in zip(obj['id'][0], obj['pos'][0]):
                        pedestrian_id = int(pedestrian_id[0][0]) - 1
                        pedestrian_pos = pedestrian_pos[0].tolist()
                        if obj_lbl[pedestrian_id] in classes and 30 < pedestrian_pos[3] <= 80:
                            class_index = classes.index(obj_lbl[pedestrian_id])
                            yolo_box_format = convertBoxFormat(pedestrian_pos, frame_size)
                            labels += f'{class_index} ' + ' '.join([str(n) for n in yolo_box_format]) + '\n'
                    if not labels:
                        continue
                    image_id = f'{set_id}_{video_id}_{frame_id}_squared'
                    label_file = open(f'{out_dir}/{image_id}.txt', 'w')
                    label_file.write(labels)
                    label_file.close()
                    print(f'Annotation for {image_id} saved')

if __name__ == '__main__':
    # 1. Конвертация .seq в .png 
    convert_seq_to_png("data/Train/set00/set00/V000.seq") 
    # 2. Преобразование .png в квадратные изображения
    squarify_images('images') 
    # 3. Конвертация .vbb в .txt
    vbb_to_txt('data/annotations/annotations/set00/V000.vbb', out_dir='labels', frame_size=(640, 640))
