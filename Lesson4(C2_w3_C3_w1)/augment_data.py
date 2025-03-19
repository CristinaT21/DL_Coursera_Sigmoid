import os
from PIL import Image, ImageEnhance
from torchvision import transforms


def augment_nested_images(
    data_dir, output_dir, brightness=0.1, contrast=0.1, degrees=10, num_augmentations=1
):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: ImageEnhance.Brightness(x).enhance(1 + brightness)
            ),  # Apply brightness
            transforms.Lambda(
                lambda x: ImageEnhance.Contrast(x).enhance(1 + contrast)
            ),  # Apply contrast
            transforms.RandomRotation(degrees=degrees),
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.0)),
        ]
    )

    # Debug: Print files in the root directory
    print(f"Files in root directory {data_dir}: {os.listdir(data_dir)}")

    for root, dirs, files in os.walk(data_dir):
        print(f"Processing directory: {root}")
        print(f"Directories: {dirs}, Files found: {files}")

        for file in files:
            file_path = os.path.join(root, file)
            label = os.path.basename(root)

            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"Skipping non-image file: {file_path}")
                continue

            try:
                print(f"Augmenting image: {file_path}")
                image = Image.open(file_path).convert("L")

                for i in range(num_augmentations):
                    augmented_image = transform(image)
                    print(
                        f"Augmented image size: {augmented_image.size}, mode: {augmented_image.mode}"
                    )

                    base_name = os.path.splitext(os.path.basename(file))[0]
                    new_filename = f"{base_name}_aug{i + 1}.jpg"
                    class_output_dir = os.path.join(output_dir, label)
                    os.makedirs(class_output_dir, exist_ok=True)
                    new_image_path = os.path.join(class_output_dir, new_filename)

                    if os.path.exists(new_image_path):
                        print(f"File already exists: {new_image_path}")
                    else:
                        try:
                            augmented_image.save(new_image_path)
                            print(f"Saved augmented image: {new_image_path}")
                        except Exception as e:
                            print(f"Error saving image: {e}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Augmented images saved in '{output_dir}'.")


# Example Usage
augment_nested_images(
    data_dir=r"C:\Users\crist\Documents\DL_Coursera_Sigmoid\Lesson4(C2_w3_C3_w1)\data_more_augmented\train",
    output_dir=r"C:\Users\crist\Documents\DL_Coursera_Sigmoid\Lesson4(C2_w3_C3_w1)\data_more_augmented\train",
    brightness=0.2,
    contrast=0.2,
    degrees=10,
    num_augmentations=3,
)
