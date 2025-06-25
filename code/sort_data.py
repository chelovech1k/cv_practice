import os
import glob
import shutil
import random


def sort_data_by_classes_limited(images_dir='images_squared', labels_dir='labels', max_per_class=300):

    class_dirs = {
        'person': {
            'images': 'sorted_data/person/images',
            'labels': 'sorted_data/person/labels'
        },
        'people': {
            'images': 'sorted_data/people/images', 
            'labels': 'sorted_data/people/labels'
        },
        'no_person': {
            'images': 'sorted_data/no_person/images',
            'labels': 'sorted_data/no_person/labels'
        }
    }

    for class_name, dirs in class_dirs.items():
        for dir_path in dirs.values():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    

    squared_images = glob.glob(f"{images_dir}/*.png")


    classified_images = {
        'person': [],
        'people': [],
        'no_person': []
    }
    
    for image_path in squared_images:
        image_name = os.path.basename(image_path)
        image_id = image_name.replace('_squared.png', '_squared')
        label_path = f"{labels_dir}/{image_id}.txt"

        image_class = classify_image(label_path)
        
        if image_class in classified_images:
            classified_images[image_class].append((image_path, label_path))

    for class_name, images in classified_images.items():
        if len(images) > max_per_class:

            random.shuffle(images)
            selected_images = images[:max_per_class]
        else:
            selected_images = images

        for image_path, label_path in selected_images:
            image_name = os.path.basename(image_path)
            
            dest_image = os.path.join(class_dirs[class_name]['images'], image_name)
            shutil.copy2(image_path, dest_image)

            if os.path.exists(label_path):
                label_name = os.path.basename(label_path)
                dest_label = os.path.join(class_dirs[class_name]['labels'], label_name)
                shutil.copy2(label_path, dest_label)


def classify_image(label_path):

    if not os.path.exists(label_path):
        return 'no_person'
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return 'no_person'

        person_count = 0
        people_count = 0
        
        for line in lines:
            line = line.strip()
            if line:
                class_id = int(line.split()[0])
                if class_id == 0:  # person
                    person_count += 1
                elif class_id == 1:  # people
                    people_count += 1

        if people_count > 0:
            return 'people'
        elif person_count > 0:
            return 'person'
        else:
            return 'no_person'
            
    except Exception as e:

        return 'no_person'


if __name__ == '__main__':


    random.seed(42)
    
    sort_data_by_classes_limited()
    #sort_data_by_classes_limited('my_images', 'my_labels', 500)

    print("- sorted_data/person/ - изображения с одиночными людьми")
    print("- sorted_data/people/ - изображения с группами людей") 
    print("- sorted_data/no_person/ - изображения без людей")
