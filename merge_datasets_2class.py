import os
import shutil

OLD_DATASET = 'Damage-Package-Detection'
NEW_DATASET = 'Package-Computer-Vision'
MERGED_DATASET = 'Merged-Dataset-2class'

for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(MERGED_DATASET, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(MERGED_DATASET, split, 'labels'), exist_ok=True)

def process_dataset(source_dir, is_new=False):
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_dir, split, 'images')
        lbl_dir = os.path.join(source_dir, split, 'labels')
        
        if not os.path.exists(img_dir): continue
            
        for img_file in os.listdir(img_dir):
            prefix = "new_" if is_new else "old_"
            new_img_name = prefix + img_file
            new_lbl_name = prefix + img_file.rsplit('.', 1)[0] + '.txt'
            
            shutil.copy(os.path.join(img_dir, img_file), os.path.join(MERGED_DATASET, split, 'images', new_img_name))
            
            lbl_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    class_id = int(parts[0])
                    
                    if not is_new:
                        if class_id == 0:
                            new_class_id = 0
                        else:
                            new_class_id = 1
                    else:
                        if class_id == 0:
                            new_class_id = 1
                        else:
                            new_class_id = 0
                    
                    parts[0] = str(new_class_id)
                    new_lines.append(" ".join(parts) + "\n")
                
                with open(os.path.join(MERGED_DATASET, split, 'labels', new_lbl_name), 'w') as f:
                    f.writelines(new_lines)

print("Объединяем датасеты в 2 класса...")
process_dataset(OLD_DATASET, is_new=False)
process_dataset(NEW_DATASET, is_new=True)
print("Готово! Папка Merged-Dataset-2class")