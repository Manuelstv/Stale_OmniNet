import os
import xml.etree.ElementTree as ET
import shutil


def count_objects_in_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return len(root.findall('.//object'))

def find_files_with_many_objects(directory, object_threshold=100):
    files_with_many_objects = []
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            file_path = os.path.join(directory, file)
            object_count = count_objects_in_xml(file_path)
            if object_count > object_threshold:
                files_with_many_objects.append(file)
    return files_with_many_objects

def move_files_with_many_objects(source_dir, target_dir, threshold=50):
    """ Move XML and corresponding JPEG files with more than 'threshold' objects. """
    for filename in os.listdir(source_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(source_dir, filename)
            num_objects = count_objects_in_xml(file_path)

            if num_objects > threshold:
                # Move XML file
                shutil.move(file_path, os.path.join(target_dir, filename))

                # Move corresponding JPEG file
                jpg_filename = filename.replace('.xml', '.jpg')
                jpg_file_path = os.path.join('/home/mstveras/ssd-360/dataset/train/images', jpg_filename)
                if os.path.exists(jpg_file_path):
                    shutil.move(jpg_file_path, os.path.join(target_dir, jpg_filename))

# Replace these with your source and target directory paths
source_directory = '/home/mstveras/ssd-360/dataset/train/labels'
target_directory = '/home/mstveras/ssd-360'

move_files_with_many_objects(source_directory, target_directory)
# Find and print the files with more than 100 objects
#files = find_files_with_many_objects(directory)
#for file in files:
#    print(file)