import os
import json
import re
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor, AutoModel, SwinModel
from loguru import logger

class MRGFieldParser:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = cfg.data.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(cfg.model.visual_encoder.hf_backbone)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iuxray":
            report_cleaner = (
                lambda t: t.replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .strip()
                .lower()
                .split(". ")
            )
            sent_cleaner = lambda t: re.sub(
                "[.,?;*!%^&_+():-\[\]{}]", "", t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower()
            )
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = " . ".join(tokens) + " ."
        # clean MIMIC-CXR reports
        else:
            report_cleaner = (
                lambda t: t.replace("\n", " ")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .replace(":", " :")
                .strip()
                .lower()
                .split(". ")
            )
            sent_cleaner = lambda t: re.sub(
                "[.,?;*!%^&_+()\[\]{}]", "", t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower()
            )
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = " . ".join(tokens) + " ."
        # report = ' '.join(report.split()[:self.cfg.max_txt_len])
        return report

    def parse(self, features):
        to_return = {"id": features["id"]}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return["input_text"] = report
        # chest x-ray images
        images = []
        for image_path in features["image_path"]:
            with Image.open(os.path.join(self.cfg.data.image_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseMRGDataset(data.Dataset):
    def __init__(self, cfg, split="train", client_idx=None):
        self.cfg = cfg
        self.meta = json.load(open(cfg.data.ann_path, "r"))
        self.meta = self.meta[split]

        if client_idx is not None and split != "test":
            data_ids = list(np.arange(len(self.meta)))

            if cfg.data.niid:
                with open(cfg.data.niid_pkl.format(cfg.train.num_clients, cfg.data.niid_type), "rb") as f:
                    client_data_ids = pickle.load(f)
                data_ids = client_data_ids[split][client_idx]
            else:
                data_ids = np.array_split(data_ids, cfg.train.num_clients)[client_idx]
            # data_ids = [i * 10 + client_idx for i in range(8)] # for debugging
            self.meta = [self.meta[idx] for idx in data_ids]
            logger.debug(f"Client {client_idx} has {len(self.meta)} samples in {split} set")

        self.parser = MRGFieldParser(cfg)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


class MVQAFieldParser:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = cfg.data.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(cfg.model.visual_encoder.hf_backbone)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        return str(report).lower()

    def parse(self, item):
        to_return = {"id": item["id"]}
        question = item.get("question", "")
        question = self.clean_report(question)
        answer = item.get("answer", "")
        answer = self.clean_report(answer)
        to_return["question"] = question
        to_return["answer"] = answer

        image_path = item["image_path"]
        with Image.open(image_path) as pil:
            array = np.array(pil, dtype=np.uint8)
            if array.shape[-1] != 3 or len(array.shape) != 3:
                array = np.array(pil.convert("RGB"), dtype=np.uint8)
            image = self._parse_image(array)

        to_return["image"] = [image]
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseMVQADataset(data.Dataset):
    def __init__(self, cfg, split="train", client_idx=None):
        self.cfg = cfg
        self.meta = json.load(open(cfg.data.ann_path, "r"))
        self.meta = self.meta[split]

        if client_idx is not None and split != "test":
            data_ids = list(np.arange(len(self.meta)))

            if cfg.data.niid:
                with open(cfg.data.niid_pkl.format(cfg.train.num_clients, cfg.data.niid_type), "rb") as f:
                    client_data_ids = pickle.load(f)
                data_ids = client_data_ids[split][client_idx]
            else:
                data_ids = np.array_split(data_ids, cfg.train.num_clients)[client_idx]
            # data_ids = [i * 10 + client_idx for i in range(8)] # for debugging
            self.meta = [self.meta[idx] for idx in data_ids]
            logger.debug(f"Client {client_idx} has {len(self.meta)} samples in {split} set")

        self.parser = MVQAFieldParser(cfg)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(cfg, client_idx=None):
    if cfg.data.dataset == "mimic-cxr" or cfg.data.dataset == "iuxray":
        ParseDataset = ParseMRGDataset
    else:
        ParseDataset = ParseMVQADataset

    train_dataset = ParseDataset(cfg, "train", client_idx=client_idx)
    dev_dataset = ParseDataset(cfg, "val", client_idx=client_idx)
    test_dataset = ParseDataset(cfg, "test", client_idx=client_idx)
    return train_dataset, dev_dataset, test_dataset


def partation_iuxray():
    import ezkfg as ez
    cfg = ez.load("config/mrg_config.yaml")
    np.random.seed(cfg.seed)

    pkl_path = f"data/iuxray-niid-{cfg.train.num_clients}-quantity.pkl"

    client_data_ids = {}

    for split in ["train", "val", "test"]:
        meta = json.load(open(cfg.data.ann_path, "r"))
        meta = meta[split]

        num_clients = cfg.train.num_clients
        niid_alpha = cfg.data.niid_alpha
        # non-iid spliting data with niid_alpha using dirichlet distribution
        data_ids = np.random.permutation(len(meta))

        print(f"Split: {split}")

        client_data_ids[split] = {}

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(niid_alpha, num_clients))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(data_ids))
        proportions = (np.cumsum(proportions) * len(data_ids)).astype(int)[:-1]
        batch_idxs = np.split(data_ids, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

        for i in range(num_clients):
            client_data_ids[i] = net_dataidx_map[i].tolist()
            print(f"Client {i} has {len(client_data_ids[i])} samples, {client_data_ids[i]}")

    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(client_data_ids, f)

    ve = SwinModel.from_pretrained(cfg.model.visual_encoder.hf_backbone)
    ip = AutoImageProcessor.from_pretrained(cfg.model.visual_encoder.hf_backbone)

    pkl_path = f"data/iuxray-niid-{cfg.train.num_clients}-feature.pkl"

    from sklearn.cluster import KMeans
    import torch
    from tqdm import tqdm

    min_size = 0
    min_require_size = 10

    client_data_ids = {}
    K = cfg.train.cross_fedproto.k

    for split in ["train", "val", "test"]:
        meta = json.load(open(cfg.data.ann_path, "r"))
        meta = meta[split]

        N = len(meta)

        num_clients = cfg.train.num_clients
        niid_alpha = cfg.data.niid_alpha
        # non-iid spliting data with niid_alpha using dirichlet distribution
        data_ids = np.random.permutation(len(meta))

        print(f"Split: {split}")

        client_data_ids[split] = {}

        feats = []
        for i in tqdm(range(len(meta))):
            img = Image.open(os.path.join(cfg.data.image_dir, meta[i]["image_path"][0]))
            img = ip(img, return_tensors="pt")
            with torch.no_grad():
                embs = ve(img.pixel_values).last_hidden_state.mean(1).squeeze(0)
            feats.append(embs)

        feats = torch.stack(feats).cpu().numpy()
        kmeans = KMeans(n_clusters=K, random_state=cfg.seed).fit(feats)
        y_train = kmeans.labels_

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(niid_alpha, num_clients))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            client_data_ids[j] = net_dataidx_map[j]
            print(f"Client {j} has {len(client_data_ids[j])} samples, {client_data_ids[j]}")

    with open(pkl_path, "wb") as f:
        pickle.dump(client_data_ids, f)


def partation_pathvqa():
    pass

def partation_vqamed():
    pass

def partation_vqarad():
    pass


if __name__ == "__main__":
    # partation_iuxray()
    partation_vqamed()
    partation_vqarad()
    partation_pathvqa()
