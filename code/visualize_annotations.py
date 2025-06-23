import os
import cv2
import glob
import numpy as np


def yolo_to_opencv_bbox(yolo_box, img_width, img_height):
    """
    Преобразует YOLO формат bbox (x_center, y_center, width, height) в OpenCV формат (x_min, y_min, x_max, y_max)
    """
    x_center, y_center, width, height = yolo_box
    
    # Преобразуем нормализованные координаты в пиксели
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Вычисляем координаты левого верхнего и правого нижнего углов
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return x_min, y_min, x_max, y_max


def draw_annotations(image_path, annotation_path, output_dir='visualized', class_names=['person', 'people']):
    """
    Рисует аннотации на изображении и сохраняет результат
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Проверяем существование файла аннотации
    if not os.path.exists(annotation_path):
        print(f"Файл аннотации не найден: {annotation_path}")
        return
    
    # Загружаем аннотации из файла
    try:
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
    except Exception as e:
        print(f"Ошибка чтения файла аннотации {annotation_path}: {e}")
        return
    
    # Цвета для разных классов (в формате BGR)
    colors = [(0, 0, 255), (0, 255, 0)]  # красный для 'person', зеленый для 'people'
    
    # Рисуем боксы для каждой аннотации
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
        
        # Преобразуем формат bbox
        x_min, y_min, x_max, y_max = yolo_to_opencv_bbox(
            [x_center, y_center, bbox_width, bbox_height], width, height
        )
        
        # Ограничиваем значения координат размерами изображения
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)
        
        # Рисуем bbox и метку класса
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Добавляем метку класса
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name}"
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Сохраняем изображение с аннотацией
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Сохранено изображение с аннотациями: {output_path}")


def visualize_dataset(images_dir='images', labels_dir='labels', output_dir='visualized', class_names=['person', 'people']):
    """
    Визуализирует все аннотации в наборе данных
    """
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Директория изображений или аннотаций не существует")
        return
    
    # Создаем директорию для выходных файлов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем все файлы аннотаций
    label_files = sorted(glob.glob(f"{labels_dir}/*.txt"))
    total = len(label_files)
    processed = 0
    failures = 0
    
    print(f"Найдено {total} файлов аннотаций")
    
    for label_file in label_files:
        # Извлекаем базовое имя файла (без расширения)
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        
        # Преобразуем имя аннотации в имя изображения (train00 -> set00)
        image_name = base_name.replace('train', 'set')
        
        # Ищем соответствующее изображение
        image_files = glob.glob(f"{images_dir}/{image_name}.*")
        
        if not image_files:
            print(f"Не найдено изображение для аннотации: {label_file}")
            failures += 1
            continue
        
        # Берем первое найденное изображение
        image_path = image_files[0]
        
        try:
            # Визуализируем
            draw_annotations(image_path, label_file, output_dir, class_names)
            processed += 1
            
            # Показываем прогресс каждые 10 изображений
            if processed % 10 == 0:
                print(f"Обработано {processed}/{total} изображений")
        except Exception as e:
            print(f"Ошибка при обработке {label_file}: {e}")
            failures += 1
    
    print(f"Обработка завершена. Успешно визуализировано {processed} из {total} изображений. Ошибок: {failures}")


def visualize_single_annotation(ann_file, img_dir='images', output_dir='visualized', class_names=['person', 'people']):
    """
    Визуализирует одну конкретную аннотацию
    """    # Получаем соответствующее изображение
    base_name = os.path.splitext(os.path.basename(ann_file))[0]
    
    # Преобразуем имя аннотации в имя изображения (train00 -> set00)
    image_name = base_name.replace('train', 'set')
    
    # Ищем соответствующее изображение
    image_files = glob.glob(f"{img_dir}/{image_name}.*")
    
    if not image_files:
        print(f"Не найдено изображение для аннотации: {ann_file}")
        return
    
    # Берем первое найденное изображение
    image_path = image_files[0]
    
    # Визуализируем
    draw_annotations(image_path, ann_file, output_dir, class_names)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Визуализация аннотаций в формате YOLO для датасета Caltech Pedestrian')
    parser.add_argument('--mode', choices=['all', 'single'], default='all',
                        help='Режим работы: all - визуализация всего датасета, single - визуализация одной аннотации')
    parser.add_argument('--annotation', help='Путь к файлу аннотации (для режима single)')
    parser.add_argument('--images-dir', default='images',
                        help='Директория с изображениями')
    parser.add_argument('--labels-dir', default='labels',
                        help='Директория с аннотациями')
    parser.add_argument('--output-dir', default='visualized',
                        help='Директория для сохранения визуализаций')
    parser.add_argument('--class-names', default='person,people',
                        help='Имена классов, разделенные запятыми')
    
    args = parser.parse_args()
    
    # Разбираем имена классов
    class_names = args.class_names.split(',')
    
    if args.mode == 'all':
        print(f"Визуализация всех аннотаций из {args.labels_dir} для изображений из {args.images_dir}")
        visualize_dataset(images_dir=args.images_dir, 
                          labels_dir=args.labels_dir, 
                          output_dir=args.output_dir,
                          class_names=class_names)
    else:  # single mode
        if args.annotation:
            print(f"Визуализация аннотации {args.annotation}")
            visualize_single_annotation(args.annotation, 
                                       img_dir=args.images_dir, 
                                       output_dir=args.output_dir,
                                       class_names=class_names)
        else:
            # Для демонстрации берем первую аннотацию из директории labels
            label_files = glob.glob(f'{args.labels_dir}/*.txt')
            if label_files:
                print(f"Аннотация не указана, использую первую найденную: {label_files[0]}")
                visualize_single_annotation(label_files[0], 
                                          img_dir=args.images_dir, 
                                          output_dir=args.output_dir,
                                          class_names=class_names)
            else:
                print(f"Не найдено файлов аннотаций в директории '{args.labels_dir}'")
                print("Укажите путь к аннотации через параметр --annotation")
