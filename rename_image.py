import os

folder_path = 'E:/DIP/Dog-Classifier/data/german_shepherd'
new_name = 'german_shepherd'  # ตั้งชื่อใหม่

for index, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        new_filename = f"{new_name}_{index+1}{os.path.splitext(filename)[1]}"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print("เปลี่ยนชื่อเสร็จแล้ว!")

# folder_path = 'E:/DIP/Dog-Classifier/data/Bangkaew'
# new_name = 'bangkaew' 

# for index, filename in enumerate(os.listdir(folder_path)):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         new_filename = f"{new_name}_{index+1}{os.path.splitext(filename)[1]}"
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# print("เปลี่ยนชื่อเสร็จแล้ว!")
