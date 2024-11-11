import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import csv
import cv2
import json
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer

# from tqdm import tqdm_notebook as tqdm
import warnings

warnings.filterwarnings("ignore")

from loguru import logger


def prepare_iuxray():
    origin_path = "data/iu_xray/NLMCXR_png"
    target_path = "data/iu_xray/NLMCXR_png_folder"

    counter_dict = {}

    os.system(f"rm -rf {target_path}")
    os.makedirs(target_path, exist_ok=True)

    for root, dirs, files in os.walk(origin_path):
        for file in files:
            if file.endswith(".png"):
                image_name = file.split(".")[0]
                image_folder = "-".join(image_name.split("-")[:-1])
                counter_dict[image_folder] = counter_dict.get(image_folder, 0) + 1
                os.makedirs(os.path.join(target_path, image_folder), exist_ok=True)
                os.system(f"cp {os.path.join(root, file)} {os.path.join(target_path, image_folder, str(counter_dict[image_folder]-1)+'.png')}")


def prepare_iuxray_from_scratch():
    data_path = "data/iu_xray/"
    logger.debug("Loading data from: {}".format(data_path))

    columns = ["image_id", "caption", "comparison", "indication", "findings", "impression", "height", "width"]
    df = pd.DataFrame(columns=columns)

    os.chdir(data_path)
    data_path = os.getcwd()
    logger.debug("Current working directory: {}".format(os.getcwd()))

    for file in tqdm(os.listdir("NLMCXR_reports/")):
        # find files ends with .xml only
        if file.endswith(".xml"):
            k = "NLMCXR_reports/"
            path = k + file
            mytree = ET.parse(path)  # parsing xml report
            comparision = mytree.find(".//AbstractText[@Label='COMPARISON']").text  # extracting comaparison text
            indication = mytree.find(".//AbstractText[@Label='INDICATION']").text  # extracting indication text
            findings = mytree.find(".//AbstractText[@Label='FINDINGS']").text  # extracting findings text
            impression = mytree.find(".//AbstractText[@Label='IMPRESSION']").text  # extracting impression text

            mytree = ET.parse(path)
            for x in mytree.findall("parentImage"):
                image_id = x.attrib["id"] + ".png"
                # print(image_id)
                filename = "NLMCXR_png/" + image_id
                image = cv2.imread(filename)  # reading image

                height, width, channels = image.shape
                caption = "" if x.find("caption").text is None else x.find("caption").text

                # df = df.append(pd.Series([image_id, caption, comparision, indication, findings, impression,height,width],
                #  index = columns), ignore_index = True)
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [[image_id, caption, comparision, indication, findings, impression, height, width]],
                            columns=columns,
                        ),
                    ],
                    ignore_index=True,
                )

    logger.debug("Data loaded successfully")
    logger.debug("Data shape: {}".format(df.shape))
    logger.debug("Data columns: {}".format(df.columns))
    logger.debug("Data head: {}".format(df.head()))
    logger.debug("Data description: {}".format(df.describe()))
    logger.debug("Data info: {}".format(df.info()))

    df.to_csv(os.path.join(data_path, "data.csv"), index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    df = pd.read_csv(os.path.join(data_path, "data.csv"))

    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Missing_Percentage"])
    missing_train_data.head(6)

    df["caption"] = df["caption"].fillna("no caption")  # filling nan with no caption
    df["comparison"] = df["comparison"].fillna("no comparison")  # filling nan with no comparison
    df["findings"] = df["findings"].fillna("no findings")  # filliing nan with no findings
    df["indication"] = df["indication"].fillna("no indication")  # filling nan with no indication
    df["impression"] = df["impression"].fillna("no impression")  # filling nan with no impression

    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Missing_Percentage"])
    missing_train_data.head(6)

    for i in df.columns[1:-2]:
        df[i] = preprocessing(df[i])

    df.replace("", float("NaN"), inplace=True)
    image_grp = []
    for i in range(df.shape[0]):
        image_grp.append("_".join(df["image_id"][i].split("-")[0:-1]))
    df["image_grp"] = image_grp

    image_per_patient = df.groupby("image_grp")["image_id"].nunique()
    logger.debug("Image per patient: {}".format(image_per_patient))

    logger.debug("Data shape: {}".format(df.shape))

    # add projection column

    df["projection"] = "PA"

    projection_df = pd.read_csv(os.path.join(data_path, "indiana_projections.csv"))

    # remove .dcm from the filename
    projection_df["filename"] = projection_df["filename"].apply(lambda x: x.split(".")[0] + ".png")

    for i in range(len(projection_df)):
        # df image_id ends with projection_df filename
        df.loc[df["image_id"].str.endswith(projection_df["filename"].iloc[i]), "projection"] = projection_df["projection"].iloc[i]

    df = df.sort_values(by="image_id").reset_index(drop=True)

    df.to_csv(os.path.join(data_path, "data_proj.csv"), index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    frontal_df = df[df["projection"] == "Frontal"]
    logger.debug("Frontal data shape: {}".format(frontal_df.shape))
    logger.debug("Frontal data head: {}".format(frontal_df.head()))

    lateral_df = df[df["projection"] == "Lateral"]
    logger.debug("Lateral data shape: {}".format(lateral_df.shape))
    logger.debug("Lateral data head: {}".format(lateral_df.head()))

    image_list = []
    for i, ind in zip(frontal_df["image_grp"], frontal_df["image_grp"].index):
        k = lateral_df[lateral_df["image_grp"] == i]["image_id"].values
        for j in range(len(k)):
            L = []
            L.append(frontal_df["image_id"][ind])
            L.append(k[j])
            L.append(frontal_df["indication"][ind])
            L.append(frontal_df["findings"][ind])
            L.append(frontal_df["impression"][ind])
            image_list.append(L)
        if len(k) == 0:
            L = []
            L.append(frontal_df["image_id"][ind])
            L.append(frontal_df["image_id"][ind])
            L.append(frontal_df["indication"][ind])
            L.append(frontal_df["findings"][ind])
            L.append(frontal_df["impression"][ind])
            image_list.append(L)

    new_df = pd.DataFrame(image_list, columns=["Frontal", "Lateral", "indication", "findings", "impression"])
    image_per_patient = df.groupby("image_grp")["image_id"].nunique()
    image_per_patient.columns = ["count"]
    c = pd.DataFrame(columns=["image_grp", "count"])
    c["image_grp"] = image_per_patient.keys()
    c["count"] = image_per_patient.values
    r = c[c["count"] == 1]["image_grp"].values

    logger.debug("len(r): {}".format(len(r)))

    image_list1 = []
    for i in r:
        k = lateral_df[lateral_df["image_grp"] == i]["image_id"].values
        ind = lateral_df[lateral_df["image_grp"] == i]["image_id"].index
        if len(k) != 0:
            L = []
            L.append(lateral_df["image_id"][ind].values[0])
            L.append(lateral_df["image_id"][ind].values[0])
            L.append(lateral_df["indication"][ind].values[0])
            L.append(lateral_df["findings"][ind].values[0])
            L.append(lateral_df["impression"][ind].values[0])
            image_list1.append(L)

    logger.debug("len(image_list1): {}".format(len(image_list1)))

    df1 = pd.DataFrame(image_list1, columns=["Frontal", "Lateral", "indication", "findings", "impression"])
    new_df = pd.concat([new_df, df1], axis=0)
    logger.debug("new_df shape: {}".format(new_df.shape))
    logger.debug("new_df head: {}".format(new_df.head()))

    new_df.to_csv(os.path.join(data_path, "data_final.csv"), index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    df = pd.read_csv(os.path.join(data_path, "data_final.csv"))


def preprocessing(df):
    column = []
    for txt in tqdm(df.values):
        txt = txt.lower()
        txt = re.sub(r"x-xxxx", "", txt)
        txt = re.sub(r"xxxx", "", txt)
        txt = re.sub(r"year-old", "", txt)

        # replace commas at the beginning of the sentence
        txt = re.sub(r"^\,", "", txt)

        # replace + to ","
        txt = re.sub(r"\+", ",", txt)
        # replace spaces before the punctuation
        txt = re.sub(r"\s+\,", ",", txt)
        txt = re.sub(r"\s+\.", ".", txt)
        txt = re.sub(r"\s+\!", "!", txt)
        txt = re.sub(r"\s+\?", "?", txt)
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"\s+\.", ".", txt)
        txt = re.sub(r"\s+\!", "!", txt)
        txt = re.sub(r"\s+\?", "?", txt)

        # multiple punctuations to single punctuation
        txt = re.sub(r"\.\.+", ".", txt)
        txt = re.sub(r"\?\?+", "?", txt)
        txt = re.sub(r"\!\!+", "!", txt)
        txt = re.sub(r"\,\,+", ",", txt)
        txt = re.sub(r"\.\,", ".", txt)
        txt = re.sub(r"\,\.", ".", txt)

        txt = txt.strip()
        if txt == "":
            txt = None
        column.append(txt)
    return column


def prepare_iuxray_to_json():
    csv_path = "data_final.csv"
    json_path = "data_final.json"

    csv_path = os.path.join(os.getcwd(), csv_path)
    json_path = os.path.join(os.getcwd(), json_path)

    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # split the data into train, val and test (7:1:2)
    # random split
    np.random.seed(0)
    indices = np.random.permutation(df.shape[0])
    train_indices = indices[: int(0.7 * df.shape[0])]
    val_indices = indices[int(0.7 * df.shape[0]) : int(0.8 * df.shape[0])]
    test_indices = indices[int(0.8 * df.shape[0]) :]

    data = {}

    data["train"] = []
    for i in train_indices:
        data["train"].append(
            {
                "id": "-".join(df["Frontal"][i].split("-")[:-1]),
                "image_path": [df["Frontal"][i], df["Lateral"][i]],
                "report": "impression: " + df["impression"][i].strip() + " findings: " + df["findings"][i].strip(),
                "split": "train",
            }
        )

    data["val"] = []
    for i in val_indices:
        data["val"].append(
            {
                "id": "-".join(df["Frontal"][i].split("-")[:-1]),
                "image_path": [df["Frontal"][i], df["Lateral"][i]],
                "report": "impression: " + df["impression"][i].strip() + " findings: " + df["findings"][i].strip(),
                "split": "val",
            }
        )

    data["test"] = []
    for i in test_indices:
        data["test"].append(
            {
                "id": "-".join(df["Frontal"][i].split("-")[:-1]),
                "image_path": [df["Frontal"][i], df["Lateral"][i]],
                "report": "impression: " + df["impression"][i].strip() + " findings: " + df["findings"][i].strip(),
                "split": "test",
            }
        )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def get_qa_len(data):
    # train, val, test question and answer max length = mean + 2*std
    tokenizer = LlamaTokenizer.from_pretrained("/workspace/hf/LLaMA2/llama-2-7b-chat")
    tokenizer.pad_token_id = 0
    
    for split in ["train", "val", "test"]:
        q_lens = []
        a_lens = []
        for i in data[split]:
            q_lens.append(len(tokenizer.encode(i["question"], add_special_tokens=False)))
            a_lens.append(len(tokenizer.encode(i["answer"], add_special_tokens=False)))

        q_lens = np.array(q_lens)
        a_lens = np.array(a_lens)

        print(split, q_lens.mean(), q_lens.std(), int(q_lens.mean() + 2 * q_lens.std()))
        print(split, a_lens.mean(), a_lens.std(), int(a_lens.mean() + 2 * a_lens.std()))


def prepare_pathvqa_to_json_old():
    qa_path = "data/PathVQA/data/QA_pairs_vb.json"
    img_path = "data/PathVQA/data"
    json_path = "data/PathVQA/PathVQA.json"

    with open(qa_path, "r") as f:
        qa_data = json.load(f)

    print(len(qa_data))

    # split the data into train, val and test (7:1:2)
    # random split
    np.random.seed(0)
    indices = np.random.permutation(len(qa_data))
    train_indices = indices[: int(0.7 * len(qa_data))]
    val_indices = indices[int(0.7 * len(qa_data)) : int(0.8 * len(qa_data))]
    test_indices = indices[int(0.8 * len(qa_data)) :]

    data = {}

    data["train"] = []

    for i in train_indices:
        data["train"].append(
            {
                "id": qa_data[i]["Image_ID"],
                "image_path": os.path.join(img_path, qa_data[i]["Image_ID"] + ".jpg"),
                "question": qa_data[i]["Questions"],
                "answer": qa_data[i]["Answers"],
                "split": "train",
            }
        )
    
    data["val"] = []

    for i in val_indices:
        data["val"].append(
            {
                "id": qa_data[i]["Image_ID"],
                "image_path": os.path.join(img_path, qa_data[i]["Image_ID"] + ".jpg"),
                "question": qa_data[i]["Questions"],
                "answer": qa_data[i]["Answers"],
                "split": "val",
            }
        )
    
    data["test"] = []

    for i in test_indices:
        data["test"].append(
            {
                "id": qa_data[i]["Image_ID"],
                "image_path": os.path.join(img_path, qa_data[i]["Image_ID"] + ".jpg"),
                "question": qa_data[i]["Questions"],
                "answer": qa_data[i]["Answers"],
                "split": "test",
            }
        )
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def prepare_slake_to_json():
    print("-------------------slake-------------------")
    qa_path = "data/Slake"
    img_path = "data/Slake/imgs"
    # json_path = "data/Slake/data.json"
    # json_path = "data/Slake/data_open.json"
    json_path = "data/Slake/data_closed.json"

    data = {}

    for split in ["train", "val", "test"]:
        data[split] = []
        qa_json = os.path.join(qa_path, (split if split != "val" else "validate") + ".json")
        with open(qa_json, "r") as f:
            qa_data = json.load(f)

        print(split, len(qa_data))

        for i in range(len(qa_data)):
            # print(i, qa_data[i]["answer_type"])
            # if str(qa_data[i]["answer_type"]) == "CLOSED":
            # if str(qa_data[i]["answer_type"]) == "OPEN":
            #     continue
            # if str(qa_data[i]["answer"]).lower() == "yes" or str(qa_data[i]["answer"]).lower() == "no":
            if str(qa_data[i]["answer"]).lower() != "yes" and str(qa_data[i]["answer"]).lower() != "no":
                continue

            data[split].append(
                {
                    "id": str(qa_data[i]["qid"]),
                    "image_path": os.path.join(img_path, qa_data[i]["img_name"]),
                    "question": str(qa_data[i]["question"]),
                    "answer": str(qa_data[i]["answer"]),
                    "answer_type": str(qa_data[i]["answer_type"]),
                    "split": split,
                }
            )
        print(split, len(data[split]))

    get_qa_len(data)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def prepare_pathvqa_to_json():
    qa_path = "data/PathVQA/pvqa/qas/"
    img_path = "data/PathVQA/pvqa/images"

    qa_path += "trainval_ans2label.pkl"

    with open(qa_path, "rb") as f:
        qa_data = pickle.load(f)

    print(len(qa_data))
    print(qa_data)

def prepare_vqamed_to_json():
    pass

def prepare_vqarad_to_json():
    print("-------------------vqarad-------------------")
    qa_path = "data/VQA-RAD/VQA_RAD Dataset Public.json"
    img_path = "data/VQA-RAD/VQA_RAD Image Folder"
    # json_path = "data/VQA-RAD/data.json"
    # json_path = "data/VQA-RAD/data_open.json"
    json_path = "data/VQA-RAD/data_closed.json"

    with open(qa_path, "r") as f:
        qa_data = json.load(f)

    print(len(qa_data))

    qa_data_ = []

    for i in range(len(qa_data)):
        # if str(qa_data[i]["answer"]).lower() == "yes" or str(qa_data[i]["answer"]).lower() == "no":
        if str(qa_data[i]["answer"]).lower() != "yes" and str(qa_data[i]["answer"]).lower() != "no":
            continue
        qa_data_.append(qa_data[i])

    qa_data = qa_data_

    print(len(qa_data))

    # split the data into train, val and test (7:1:2)
    # random split
    np.random.seed(0)
    indices = np.random.permutation(len(qa_data))
    train_indices = indices[: int(0.7 * len(qa_data))]
    val_indices = indices[int(0.7 * len(qa_data)) : int(0.8 * len(qa_data))]
    test_indices = indices[int(0.8 * len(qa_data)) :]

    data = {}

    data["train"] = []

    for i in train_indices:
        data["train"].append(
            {
                "id": str(qa_data[i]["qid"]),
                "image_path": os.path.join(img_path, qa_data[i]["image_name"]),
                "question": str(qa_data[i]["question"]),
                "answer": str(qa_data[i]["answer"]),
                "answer_type": str(qa_data[i]["answer_type"]),
                "split": "train",
            }
        )

    data["val"] = []

    for i in val_indices:
        data["val"].append(
            {
                "id": str(qa_data[i]["qid"]),
                "image_path": os.path.join(img_path, qa_data[i]["image_name"]),
                "question": str(qa_data[i]["question"]),
                "answer": str(qa_data[i]["answer"]),
                "answer_type": str(qa_data[i]["answer_type"]),
                "split": "val",
            }
        )

    data["test"] = []

    for i in test_indices:
        data["test"].append(
            {
                "id": str(qa_data[i]["qid"]),
                "image_path": os.path.join(img_path, qa_data[i]["image_name"]),
                "question": str(qa_data[i]["question"]),
                "answer": str(qa_data[i]["answer"]),
                "answer_type": str(qa_data[i]["answer_type"]),
                "split": "test",
            }
        )
    
    for split in ["train", "val", "test"]:
        print(split, len(data[split]))

    get_qa_len(data)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    cwd = os.getcwd()
    prepare_iuxray_from_scratch()
    prepare_iuxray_to_json()
    # prepare_pathvqa_to_json()
    os.chdir(cwd)
    prepare_vqarad_to_json()
    os.chdir(cwd)
    prepare_slake_to_json()
