import os
import xml.etree.ElementTree as ET

def remove_non_person_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate over all 'object' elements and remove those that are not 'person'
    objects = root.findall('object')
    for obj in objects:
        name = obj.find('name').text
        if name != 'person':
            root.remove(obj)

    # Save the modified XML back to the file
    tree.write(xml_file)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            xml_file = os.path.join(folder_path, filename)
            remove_non_person_boxes(xml_file)
            print(f"Processed {xml_file}")

# Replace 'your_folder_path' with the path to your folder containing XML files
process_folder('labels')