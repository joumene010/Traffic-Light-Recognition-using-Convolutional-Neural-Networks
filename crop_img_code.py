def crop_image_and_extract_label(image, label_path, output_folder, split_name):
    with open(label_path, "r") as label_file:
        lines = label_file.readlines()
        labels = []

        for i, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])  # Extract the class ID as label
            x, y, w, h = map(float, parts[1:])  # Darknet YOLO format
            width, height = image.size
            x_min = int((x - w / 2) * width)
            y_min = int((y - h / 2) * height)
            x_max = int((x + w / 2) * width)
            y_max = int((y + h / 2) * height)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            cropped_image = image.crop((x_min, y_min, x_max, y_max)).resize((90, 90))
            labels.append(class_id)

            # Save the cropped image
            output_dir = os.path.join(output_folder, split_name, str(class_id))
            os.makedirs(output_dir, exist_ok=True)
            cropped_image.save(os.path.join(output_dir, f"{os.path.basename(label_path).split('.')[0]}_{i}.png"))

        return labels

def fetch_image(data_folder, output_folder):
    splits = {
        "train": os.path.join(data_folder, "train"),
        "test": os.path.join(data_folder, "test"),
        "valid": os.path.join(data_folder, "valid")
    }

    for split_name, split_path in splits.items():
        images_folder = os.path.join(split_path, "images")
        labels_folder = os.path.join(split_path, "labels")

        for image_name in os.listdir(images_folder):
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_name)
            image_path = os.path.join(images_folder, image_name)

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {image_name}")
                continue

            imgTrafficSignal = Image.open(image_path).convert("RGB")
            crop_image_and_extract_label(imgTrafficSignal, label_path, output_folder, split_name)

# Data folder
data_folder = "./DataSet"
output_folder = "./CroppedDataSet"
fetch_image(data_folder, output_folder)