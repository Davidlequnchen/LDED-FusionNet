import csv
import os


Dataset = '/home/chenlequn/Dataset/LDED_acoustic_visual_monitoring_dataset/segmented_25Hz/23'

input_csv = os.path.join(Dataset, 'annotations_23.csv')
output_txt = os.path.join(Dataset, 'annotations_23.txt') 

class_colors = {
    'Crack': 'red',
    'Keyhole pores': 'green',
    'Laser-off': 'blue',
    'Defect-free': 'yellow',
}

class_codes = {
    'Crack': 1,
    'Keyhole pores': 2,
    'Laser-off': 3,
    'Defect-free': 4,
}


with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as txt_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip the first row (header)
    
    for row in reader:
        sample_index, audio_name, image_name, class_name, class_ID, class_name_2, class_ID_2, layer_number, sample_number = row
        # Calculate the frame's timestamp 25 fps
        timestamp = int(sample_index) / 25
        # color = class_colors[class_name]
        # class_code = class_codes[class_name]
        txt_file.write(f"{timestamp}\t{class_name}\t{class_name_2}\t{layer_number}\n")
