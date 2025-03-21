from pathlib import Path
import json
import shutil
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Muret JSON files in nested folders to a list of JSON files in a single folder")
    parser.add_argument(
        "muret_folder",
        type=str,
        help="Path to the folder containing the Muret JSON files"
    )
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="Path where the processed dataset will be stored"
    )
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    muret_folder = Path(args.muret_folder)
    dataset_folder = Path(args.dataset_folder)

    # Create the dataset folder if it does not exist
    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)

    # Iterate over all json files in the folder if they are not called document.json
    for file in tqdm(muret_folder.rglob("*.json")):
        if file.stem == "document":
            continue
        # read the content of the json file
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # add the musicXML information from the documen.json in the same folder
        with open(file.parent / "document.json", "r", encoding="utf-8") as f:
            document = json.load(f)
        # next lines are a bit weird, but necessary, since document["encodings"] and data["encodings"] are lists of dictionaries with only one key
        kern = [e for e in data["encodings"] if list(e.keys())[0] == "**kern"][0]["**kern"]
        musicxml = [e for e in document["encodings"] if list(e.keys())[0] == "MusicXML"][0]["MusicXML"]
        # rewrite "encodings" field from the json file
        data["encodings"] = {"**kern": kern, "musicxml": musicxml}
        # change the image url to the muret server
        data["original"] = data["original"].replace("http://localhost:8182/", "https://muret.dlsi.ua.es/images/")

        # the new name of the json file is taken from the "name" field of the json file
        new_name = data["name"]
        # correct quality problems in the name
        new_name = correct_name(new_name)
        data["name"] = new_name

        # add the extension
        new_name += ".json"

        # save the new json file in the dataset folder
        with open(dataset_folder / new_name, "w", encoding="utf-8") as f:
            json.dump(data, f)

def correct_name(name):
    # if name contains an extension (for example .jpeg) it was a mistake. Remove it
    if len(name.split(".")) != 1:
        print("Problem with the name of the file: ", name)
        print(f"The extension {name.split('.')[1]} will be removed")
        name = name.split(".")[0]

    # if name (excluding version) contains "_" it was a mistake. Replace it with "-"
    has_version = len(name.split("_version_")) == 2
    name_without_version = name.split("_version_")[0] if has_version else name
    if "_" in name_without_version:
        print("Problem with the name of the file: ", name)
        print(f"The underscore will be replaced by -")
        name_without_version = name_without_version.replace("_", "-")
    if has_version:
        name = name_without_version + "_version_" + name.split("_version_")[1]
    else:
        name = name_without_version
    
    # if dont is a single word, it was a mistake, separate it with "-"
    if "dont" in name:
        print("Problem with the name of the file: ", name)
        print(f"The word dont will be separated by -")
        name = name.replace("dont", "don-t")

    # if name contains "--" it was a mistake. Replace it with "-"
    if "--" in name:
        print("Problem with the name of the file: ", name)
        print(f"The double dash will be replaced by -")
        name = name.replace("--", "-")
    
    return name


if __name__ == "__main__":
    main()


