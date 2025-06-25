import os
import sys
sys.path.append('.')
from preprocessing import convert_seq_to_png, squarify_images, vbb_to_txt


def create_test_data():

    base_dirs = ['images', 'images_squared', 'labels']
    for dir_name in base_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    test_files = [
        {
            'seq': '../data/Train/set00/set00/V000.seq',  
            'vbb': '../data/annotations/annotations/set00/V000.vbb',
            'description': 'set00/V000 - содержит person и people'
        },
        {
            'seq': '../data/Train/set02/set02/V000.seq',
            'vbb': '../data/annotations/annotations/set02/V000.vbb', 
            'description': 'set02/V000 - нет людей'
        }
    ]
    
    for i, test_file in enumerate(test_files):
        print(f"\n=== Обработка тестового файла {i+1}: {test_file['description']} ===")

        if not os.path.exists(test_file['seq']):
            continue
            
        if not os.path.exists(test_file['vbb']):
            continue

        convert_seq_to_png(test_file['seq'], 'images')
        
        squarify_images('images', 'images_squared')

        vbb_to_txt(test_file['vbb'], 'labels', frame_size=(640, 640))

    check_created_data()


def check_created_data():

    if os.path.exists('labels'):
        person_files = []
        people_files = []
        empty_files = []
        
        for label_file in os.listdir('labels'):
            if label_file.endswith('.txt'):
                label_path = os.path.join('labels', label_file)
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    
                    if not content:
                        empty_files.append(label_file)
                    else:
                        lines = content.split('\n')
                        has_person = any(line.startswith('0 ') for line in lines)
                        has_people = any(line.startswith('1 ') for line in lines)
                        
                        if has_people:
                            people_files.append(label_file)
                        elif has_person:
                            person_files.append(label_file)
                        else:
                            empty_files.append(label_file)
                            
                except Exception as e:
                    print(f"   ⚠️ Ошибка чтения {label_file}: {e}")
        



def test_sorting_with_real_data():

    from sort_data import sort_data_by_classes_limited
    

    import shutil
    if os.path.exists('sorted_data'):
        shutil.rmtree('sorted_data')

    sort_data_by_classes_limited(
        images_dir='images_squared',
        labels_dir='labels', 
        max_per_class=50  # Ограничиваем для теста
    )

    total_images = 0
    total_labels = 0
    
    for class_name in ['person', 'people', 'no_person']:
        images_dir = f'sorted_data/{class_name}/images'
        labels_dir = f'sorted_data/{class_name}/labels'
        
        if os.path.exists(images_dir):
            images_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
            labels_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')]) if os.path.exists(labels_dir) else 0
            print(f"   {class_name}: {images_count} изображений, {labels_count} аннотаций")
            total_images += images_count
            total_labels += labels_count
        else:
            print(f"   {class_name}: папка не создана")
    print(f"\nВсего: {total_images} изображений, {total_labels} аннотаций")

if __name__ == '__main__':

    try:

        create_test_data()

        test_sorting_with_real_data()

        
    except Exception as e:

        import traceback
        traceback.print_exc()
