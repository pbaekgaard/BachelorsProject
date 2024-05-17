import os
import shutil
import random


def sanitize_data(data):
    # Remove any semicolons from the data
    sanitized_data = [entry.replace(";", "") for entry in data]
    return sanitized_data


def merge_data(folder, labels):
    # Create a dictionary to store data for each user
    user_data = {}
    output_folder = folder
    accel_folder = "raw/phone/accel"
    gyro_folder = "raw/phone/gyro"

    # REMOVE FOLDER IF IT ALREADY EXISTS
    if os.path.exists(output_folder):
        shutil.rmtree(
            output_folder,
        )

    # Process accelerometer data
    for accel_file in os.listdir(accel_folder):
        if accel_file.startswith("data") and accel_file.endswith("_accel_phone.txt"):
            user_id = accel_file.split("_")[1]
            with open(os.path.join(accel_folder, accel_file), "r") as file:
                for line in file:
                    parts = line.strip().split(",")
                    activity_label = parts[1]
                    if activity_label not in labels:
                        continue
                    timestamp = parts[2]
                    accel_data = sanitize_data(parts[3:])

                    if user_id not in user_data:
                        user_data[user_id] = {"accel": [], "gyro": []}

                    user_data[user_id]["accel"].append((activity_label, timestamp, accel_data))

    # Process gyroscope data
    for gyro_file in os.listdir(gyro_folder):
        if gyro_file.startswith("data") and gyro_file.endswith("_gyro_phone.txt"):
            user_id = gyro_file.split("_")[1]
            with open(os.path.join(gyro_folder, gyro_file), "r") as file:
                for line in file:
                    parts = line.strip().split(",")
                    activity_label = parts[1]
                    if activity_label not in labels:
                        continue
                    timestamp = parts[2]
                    gyro_data = sanitize_data(parts[3:])

                    if user_id in user_data:
                        user_data[user_id]["gyro"].append((timestamp, gyro_data))

    # Write merged data to output files
    output_file_path = output_folder
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for user_id, data in user_data.items():
        # Merge data and write to output file
        with open(os.path.join(output_file_path, f"data_{user_id}_merged.csv"), "w") as output_file:
            output_file.write("Subject-Id,Activity Label,Time stamp,Accel_x,Accel_y,Accel_z,Gyro_x,Gyro_y,Gyro_z\n")
            for accel_entry, gyro_entry in zip(data["accel"], data["gyro"]):
                accel_data = ",".join(accel_entry[2])
                gyro_data = ",".join(gyro_entry[1])
                output_file.write(f"{user_id},{accel_entry[0]},{accel_entry[1]},{accel_data},{gyro_data}\n")

    # Split files into Test, Training, and Validation sets
    output_files = os.listdir(output_file_path)
    random.shuffle(output_files)
    num_files = len(output_files)
    test_files = int(num_files * 0.15)
    train_files = int(num_files * 0.8)

    valid_files = num_files - test_files - train_files

    test_set = output_files[:test_files]
    train_set = output_files[test_files : test_files + train_files]
    valid_set = output_files[test_files + train_files :]

    test_folder = os.path.join(output_file_path, "Test")
    train_folder = os.path.join(output_file_path, "Training")
    valid_folder = os.path.join(output_file_path, "Validation")

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)

    for file in test_set:
        shutil.move(os.path.join(output_file_path, file), test_folder)

    for file in train_set:
        shutil.move(os.path.join(output_file_path, file), train_folder)

    for file in valid_set:
        shutil.move(os.path.join(output_file_path, file), valid_folder)


print("Generating NewData...")
merge_data("NewData", ["C"])
print("Generating ProcessedData...")
merge_data("ProcessedData", ["A", "B", "C"])
print("Generating StupidTestData...")
merge_data("StupidTestData", ["A", "B", "C"])
print("Done!")
