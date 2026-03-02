total_images = 0

for class_name in classes:
    total_images += len(os.listdir(os.path.join(train_dir, class_name)))

print("Total Training Images:", total_images)
print("Number of Classes:", len(classes))