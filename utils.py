import torch
import numpy as np
from loguru import logger
import ezkfg as ez
from dataset.data_module import DataModule
from model.mrg import MRGModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def get_mean_iuxray(mean_type="token", cfg=None, global_params=None):  # "token, original"
    dm = DataModule(cfg=cfg)
    dm.setup("")

    mean_frontal, mean_lateral = None, None
    mean_text = None

    if mean_type == "token": 
        model = MRGModel(cfg)
        if global_params is not None:
            model.set_trainable_params(global_params)

        model = model.to("cuda")

        for batch in dm.train_dataloader():
            with torch.no_grad():
                batch["image"] = [img.to("cuda") for img in batch["image"]]
                _, _, img_embs1 = model.encode_img([batch["image"][0]], return_feats=True)
                _, _, img_embs2 = model.encode_img([batch["image"][1]], return_feats=True)
                text = [txt + cfg.model.tokenizer.eos_token for txt in batch["input_text"]]

                model.tokenizer.padding_side = "right"
                text_token = model.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=cfg.model.max_length, add_special_tokens=False
                )["input_ids"]
                text_token = text_token.to("cuda")
                text_embs = model.embed_tokens(text_token)

            # logger.debug(f"img_embs[0].shape: {img_embs[0].shape}")
            # logger.debug(f"img_embs[1].shape: {img_embs[1].shape}")
            # logger.debug(f"text_embs.shape: {text_embs.shape}")

            # type
            # logger.debug(f"img_embs[0].dtype: {img_embs[0].dtype}")
            # logger.debug(f"img_embs[1].dtype: {img_embs[1].dtype}")
            # logger.debug(f"text_embs.dtype: {text_embs.dtype}")

            # logger.debug(f"type(img_embs[0]): {type(img_embs[0])}")
            # logger.debug(f"type(img_embs[1]): {type(img_embs[1])}")
            # logger.debug(f"type(text_embs): {type(text_embs)}")

            if mean_frontal is None:
                mean_frontal = img_embs1.cpu().detach().numpy().sum(axis=0)
                mean_lateral = img_embs2.cpu().detach().numpy().sum(axis=0)
                mean_text = text_embs.cpu().detach().numpy()
            else:
                mean_frontal = mean_frontal + img_embs1.cpu().detach().numpy().sum(axis=0)
                mean_lateral = mean_lateral + img_embs2.cpu().detach().numpy().sum(axis=0)
                mean_text = mean_text + text_embs.cpu().detach().numpy()

        mean_frontal /= len(dm.dataset["train"])
        mean_lateral /= len(dm.dataset["train"])
        mean_text /= len(dm.dataset["train"])
        mean_text = mean_text.mean(axis=0)
        return (mean_frontal, mean_lateral, mean_text)

    elif mean_type == "original":

        tokenizer = LlamaTokenizer.from_pretrained(cfg.model.text_decoder.hf_backbone)
        tokenizer.pad_token_id = 0

        for sample in dm.dataset["train"]:
            frontal, lateral = sample["image"][0], sample["image"][1]
            text = tokenizer(
                sample["input_text"] + cfg.model.tokenizer.eos_token,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg.model.max_length,
                add_special_tokens=False,
            )["input_ids"]

            frontal = frontal.numpy()
            lateral = lateral.numpy()
            text = text.numpy()[0]

            if mean_frontal is None:
                mean_frontal = frontal
                mean_lateral = lateral
                mean_text = text
            else:
                mean_frontal += frontal
                mean_lateral += lateral
                # padding
                mean_text += text

        logger.debug(f"mean_frontal.shape: {mean_frontal.shape}")
        logger.debug(f"mean_lateral.shape: {mean_lateral.shape}")
        logger.debug(f"mean_text.shape: {mean_text.shape}")

        mean_frontal /= len(dm.dataset["train"])
        mean_lateral /= len(dm.dataset["train"])
        mean_text = mean_text / len(dm.dataset["train"])
        # force cast to int
        mean_text = mean_text.astype(np.int64)
        return (mean_frontal, mean_lateral, mean_text)
    else:
        raise ValueError(f"Invalid mean type: {mean_type}")


def get_mean_mvqa(mean_type="token", cfg=None, global_params=None):  # "token, original"
    dm = DataModule(cfg=cfg)
    dm.setup("")

    mean_img = None
    mean_text = None

    if mean_type == "token":
        model = MVQAModel(cfg)
        if global_params is not None:
            model.set_trainable_params(global_params)

        model = model.to("cuda")

        for batch in dm.train_dataloader():
            with torch.no_grad():
                batch["image"] = [img.to("cuda") for img in batch["image"]]
                _, _, img_embs = model.encode_img(batch["image"], return_feats=True)
                # print(batch["answer"])
                text = [txt + cfg.model.tokenizer.eos_token for txt in batch["answer"]]

                model.tokenizer.padding_side = "right"
                text_token = model.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=cfg.model.max_length, add_special_tokens=False
                )["input_ids"]
                text_token = text_token.to("cuda")
                text_embs = model.embed_tokens(text_token)

            # logger.debug(f"img_embs[0].shape: {img_embs[0].shape}")
            # logger.debug(f"img_embs[1].shape: {img_embs[1].shape}")
            # logger.debug(f"text_embs.shape: {text_embs.shape}")

            # type
            # logger.debug(f"img_embs[0].dtype: {img_embs[0].dtype}")
            # logger.debug(f"img_embs[1].dtype: {img_embs[1].dtype}")
            # logger.debug(f"text_embs.dtype: {text_embs.dtype}")

            # logger.debug(f"type(img_embs[0]): {type(img_embs[0])}")
            # logger.debug(f"type(img_embs[1]): {type(img_embs[1])}")
            # logger.debug(f"type(text_embs): {type(text_embs)}")

            if mean_img is None:
                mean_img = img_embs.cpu().detach().numpy().sum(axis=0)
                mean_text = text_embs.cpu().detach().numpy()
            else:
                mean_img = mean_img + img_embs.cpu().detach().numpy().sum(axis=0)
                mean_text = mean_text + text_embs.cpu().detach().numpy()

        mean_img /= len(dm.dataset["train"])
        mean_text /= len(dm.dataset["train"])
        mean_text = mean_text.mean(axis=0)
        return (mean_img, mean_text)

    elif mean_type == "original":

        tokenizer = LlamaTokenizer.from_pretrained(cfg.model.text_decoder.hf_backbone)
        tokenizer.pad_token_id = 0

        for sample in dm.dataset["train"]:
            img = sample["image"][0]
            text = tokenizer(
                sample["answer"] + cfg.model.tokenizer.eos_token,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg.model.max_length,
                add_special_tokens=False,
            )["input_ids"]

            img = img.numpy()
            text = text.numpy()[0]

            if mean_img is None:
                mean_img = img
                mean_text = text
            else:
                mean_img += img
                # padding
                mean_text += text

        logger.debug(f"mean_img.shape: {mean_img.shape}")
        logger.debug(f"mean_text.shape: {mean_text.shape}")

        mean_img /= len(dm.dataset["train"])
        mean_text = mean_text / len(dm.dataset["train"])
        # force cast to int
        mean_text = mean_text.astype(np.int64)
        return (mean_img, mean_text)
    else:
        raise ValueError(f"Invalid mean type: {mean_type}")


def test_iuxray():
    cfg = ez.load("config/mrg_config.yaml")
    mean = get_mean_iuxray(mean_type="token", cfg=cfg)
    print(mean)
    print("shape: ", mean[0].shape, mean[1].shape, mean[2].shape)
    # (49, 4096) (49, 4096) (10, 4096)
    # mean = get_mean_iuxray(mean_type="original", cfg=cfg)
    # print(mean)
    # print("shape: ", mean[0].shape, mean[1].shape, mean[2].shape)
    # (3, 224, 224) (3, 224, 224) (10,)
    print("Done!")

def test_mvqa():
    cfg = ez.load("config/mvqa_config.yaml")
    mean = get_mean_mvqa(mean_type="token", cfg=cfg)
    print(mean)
    print("shape: ", mean[0].shape, mean[1].shape)
    # (49, 4096) (60, 4096)
    mean = get_mean_mvqa(mean_type="original", cfg=cfg)
    print(mean)
    print("shape: ", mean[0].shape, mean[1].shape)
    # (3, 224, 224) (60,)
    print("Done!")

if __name__ == "__main__":
    # test_iuxray()
    test_mvqa()
