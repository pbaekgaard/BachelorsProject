import pandas as pd
import os


def initDirectories():
    # Define the main directory path
    processed_data_path = "./ProcessedData"
    # Define subfolder names (list for reusability)
    subfolders = ["Accel", "Gyro"]

    # Loop through each subfolder
    for folder in subfolders:
        # Create the main subfolder path
        folder_path = os.path.join(processed_data_path, folder)

        # Create the main subfolder (if it doesn't exist)
        os.makedirs(folder_path, exist_ok=True)

        # Define sub-subfolders (list for reusability)
        subsubfolders = ["Test", "Validation", "Training"]

        # Loop through each sub-subfolder and create them within the main subfolder
        for subfolder in subsubfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)


def readAndProcess(path, processedPath):
    dataFileArray = os.listdir(path)
    dataFileArray.pop(0)
    dataFileArray.pop(0)
    classes = ["B", "C", "M"]
    column_index = 1
    index = 0
    for file in dataFileArray:
        file_path = os.path.join(path, file)
        if ".DS_Store" not in file_path:
            df = pd.read_csv(file_path, header=None, sep=",", encoding="ISO-8859-1")
            filtered_df = df[df.iloc[:, column_index].isin(classes)]
            eightyPercent = (len(dataFileArray) / 100) * 80
            fifteenPercent = (len(dataFileArray) / 100) * 15
            # The following limits the number of rows to every (rowFrequency) number of rows
            rowFrequency = 5
            limited_df = filtered_df.iloc[::rowFrequency]
            limited_df.columns = [
                "Subject-id",
                "Activity Label",
                "Timestamp",
                "x",
                "y",
                "z",
            ]
            if index < eightyPercent:
                processedPathExtended = processedPath + "Training/"
                f = open(
                    os.path.splitext(processedPathExtended + file)[0] + ".csv", "w"
                )
                f.write(limited_df.to_csv(sep=",", index=False).replace(";", ""))
                f.close()

            elif index >= eightyPercent and index <= (eightyPercent + fifteenPercent):
                processedPathExtended = processedPath + "Test/"
                f = open(
                    os.path.splitext(processedPathExtended + file)[0] + ".csv", "w"
                )
                f.write(limited_df.to_csv(sep=",", index=False).replace(";", ""))
                f.close()
            else:
                processedPathExtended = processedPath + "Validation/"
                f = open(
                    os.path.splitext(processedPathExtended + file)[0] + ".csv", "w"
                )
                f.write(limited_df.to_csv(sep=",", index=False).replace(";", ""))
                f.close()
            index += 1


pathAccel = "./raw/phone/accel"
pathGyro = "./raw/phone/gyro"
processedPathAccel = "./ProcessedData/Accel/"
processedPathGyro = "./ProcessedData/Gyro/"

initDirectories()

readAndProcess(pathAccel, processedPathAccel)
readAndProcess(pathGyro, processedPathGyro)
