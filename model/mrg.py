import os
import json
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from evalcap.bleu.bleu import Bleu
from evalcap.meteor.meteor import Meteor
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType

from loguru import logger

from algor.prototype import get_img_proto_props, get_txt_proto_props, get_miss_loss, get_sim_loss


class MRGModel(pl.LightningModule):
    """
    R2GenGPT model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.dict())

        logger.info(f"Loading vision encoder:{cfg.model.visual_encoder}")
        self.visual_encoder = SwinModel.from_pretrained(cfg.model.visual_encoder.hf_backbone)
        if cfg.model.visual_encoder.use_lora:
            peft_config_visual = LoraConfig(
                r=cfg.model.visual_encoder.lora.r,
                lora_alpha=cfg.model.visual_encoder.lora.alpha,
                target_modules=["query", "value"],
                lora_dropout=cfg.model.visual_encoder.lora.dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            self.freeze_ve = False
            logger.info("Loading vision encoder with LoRA -- Done")
        elif cfg.model.visual_encoder.freeze:
            self.freeze_encoder()
            self.freeze_ve = True
            logger.info(f"Loading Frozen vision encoder:{cfg.model.visual_encoder.hf_backbone} -- Done")
        else:
            self.freeze_ve = False
            logger.info(f"Loading Trainable vision encoder:{cfg.model.visual_encoder.hf_backbone} -- Done")

        logger.info("Loading LLAMA")
        # self.tokenizer = LlamaTokenizer.from_pretrained(cfg.model.text_decoder.hf_backbone, use_fast=False)
        self.tokenizer = LlamaTokenizer.from_pretrained(cfg.model.text_decoder.hf_backbone)
        self.tokenizer.pad_token_id = 0
        if cfg.model.text_decoder.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading LLaMA model in 8-bit mode")
            self.model = LlamaForCausalLM.from_pretrained(
                cfg.model.text_decoder.hf_backbone, torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto"
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                cfg.model.text_decoder.hf_backbone,
                torch_dtype=torch.float16,
            )

        if cfg.model.text_decoder.use_lora:
            self.embed_tokens = self.model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=cfg.model.text_decoder.lora.r,
                lora_alpha=cfg.model.text_decoder.lora.alpha,
                lora_dropout=cfg.model.text_decoder.lora.dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            logger.info("Loading LLAMA LoRA Done")
        else:
            self.embed_tokens = self.model.get_input_embeddings()
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            logger.info("Loading LLAMA Done")

        self.proj = nn.Linear(self.visual_encoder.num_features, self.model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)

        # init project layer and layer norm
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
        nn.init.constant_(self.layer_norm.weight, 1)
        nn.init.constant_(self.layer_norm.bias, 0)

        self.eos_token = cfg.model.tokenizer.eos_token
        self.prompt = "Generate a comprehensive and detailed diagnosis report for this chest xray image."
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.stage = None
        self.automatic_optimization = True

        if cfg.model.delta_file is not None:
            state_dict = torch.load(cfg.model.delta_file, map_location=torch.device(f"cuda:{torch.cuda.current_device()}"))["model"]
            self.load_state_dict(state_dict=state_dict, strict=False)
            logger.info(f"Load checkpoint from {cfg.model.delta_file}")

    def freeze_encoder(self):
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.freeze_ve = True
        logger.info("Freezing vision encoder -- Done")

    def unfreeze_encoder(self):
        for name, param in self.visual_encoder.named_parameters():
            # if self.cfg.model.visual_encoder.use_lora:
            #     if "lora" in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            # else:
            param.requires_grad = True
        self.freeze_ve = False
        logger.info("Unfreezing vision encoder -- Done")

    def set_stage(self, stage):
        self.stage = stage

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Rouge(), "ROUGE_L"), (Meteor(), "METEOR"), (Cider(), "CIDEr")]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images, return_feats=False):
        image_embeds = []
        for image in images:
            device = image.device
            if self.cfg.model.visual_encoder.global_only:
                image_embed = self.visual_encoder(image)["pooler_output"].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)["last_hidden_state"].to(device)
            image_embeds.append(image_embed)

        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        if return_feats:
            return inputs_llama, atts_llama, image_embeds
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embs, img_atts):
        prompt = f"Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:"
        batch_size = img_embs.shape[0]
        p_before, p_after = prompt.split("<ImageHere>")
        p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embs.device)
        p_after_tokens = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embs.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embs = torch.cat([p_before_embeds, img_embs, p_after_embeds], dim=1)
        wrapped_img_atts = img_atts[:, :1].expand(-1, wrapped_img_embs.shape[1])
        return wrapped_img_embs, wrapped_img_atts

    def forward(self, samples):
        image = samples["image"]

        client_type = getattr(self.cfg, "client_type", None)
        mm_str = getattr(self.cfg.train, "mm_strategy", None)
        mm_mean_type = getattr(self.cfg.train, "mm_mean_type", None)
        miss_loss = 0.0
        sim_loss = 0.0

        # logger.debug(f"Client type: {client_type}, mm_str: {mm_str}, mm_mean_type: {mm_mean_type}")

        if self.cfg.train.modal_hetero and mm_str == "duplicate":
            # logger.debug(f"Use strategy {mm_str}")
            if client_type == 1:
                image = [image[0]] * 2
            elif client_type == 2:
                image = [image[1]] * 2
        elif self.cfg.train.modal_hetero and mm_str == "mean" and mm_mean_type == "original":
            # logger.debug(f"Use strategy {mm_str}, mean_type {mm_mean_type}, processing images")
            means = self.cfg.payload["means"]
            if client_type == 1:
                image = [image[0], torch.from_numpy(means[1]).expand(image[0].shape[0], -1, -1, -1).to(image[0].device)]
            elif client_type == 2:
                image = [torch.from_numpy(means[0]).expand(image[1].shape[0], -1, -1, -1).to(image[1].device), image[1]]

        if self.cfg.train.modal_hetero and mm_str == "mean" and mm_mean_type == "token":
            # logger.debug(f"Use strategy {mm_str}, mean_type {mm_mean_type}, processing images")
            means = self.cfg.payload["means"]
            # logger.debug(f"means[0].shape: {means[0].shape}")
            # logger.debug(f"means[1].shape: {means[1].shape}")
            # logger.debug(f"means[2].shape: {means[2].shape}")
            if client_type == 1:
                img_embs, img_atts = self.encode_img([image[0]])
                missing_embs = torch.from_numpy(means[1]).expand(img_embs.shape[0], -1, -1).to(image[0].device)
                img_embs = torch.mean(torch.stack([img_embs, self.proj(missing_embs)], dim=0), dim=0)
            elif client_type == 2:
                img_embs, img_atts = self.encode_img([image[1]])
                missing_embs = torch.from_numpy(means[0]).expand(img_embs.shape[0], -1, -1).to(image[1].device)
                img_embs = torch.mean(torch.stack([self.proj(missing_embs), img_embs], dim=0), dim=0)
            else:
                img_embs, img_atts = self.encode_img(image)
        elif self.cfg.train.modal_hetero and mm_str == "prototype" and (client_type == 1 or client_type == 2):
            # logger.debug(f"Use strategy {mm_str}, computing prototype loss")
            global_prototype = getattr(self.cfg.payload, "global_prototype", None)
            # TODO
            if global_prototype is None:
                if client_type == 1:
                    image = [image[0]] * 2
                elif client_type == 2:
                    image = [image[1]] * 2
                img_embs, img_atts = self.encode_img(image)
            else:
                def get_txt_emb():
                    self.tokenizer.padding_side = "right"
                    text = [t + self.eos_token for t in samples["input_text"]]

                    to_regress_tokens = self.tokenizer(
                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.cfg.model.max_length, add_special_tokens=False
                    ).to(image[0].device)

                    to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
                    return to_regress_embeds

                txt_embs = get_txt_emb()
                img_protos = global_prototype[0]
                txt_protos = global_prototype[1]
                cross_matrix = global_prototype[2]

                txt_proto_props, txt_proto_ids = get_txt_proto_props(txt_embs, txt_protos)
                batch_size, txt_seq_len, txt_embed_dim = txt_embs.shape

                img_proto_weight = np.array([img_proto[2] for img_proto in img_protos])

                if client_type == 1:
                    img_embs, img_atts, img_feats = self.encode_img([image[0]], return_feats=True)
                elif client_type == 2:
                    img_embs, img_atts, img_feats = self.encode_img([image[1]], return_feats=True)

                _, img_seq_len, img_embed_dim = img_embs.shape

                miss_loss += get_miss_loss(img_feats, img_protos)

                miss_embs = []
                txt_proto_props = txt_proto_props.reshape(batch_size, txt_seq_len, -1)
                txt_proto_ids = txt_proto_ids.reshape(batch_size, -1)
                img_proto_ids = []

                for bs in range(batch_size):
                    miss_emb_prob = torch.zeros(len(global_prototype[0]))

                    for i in range(txt_seq_len):
                        miss_emb_prob += cross_matrix[:, txt_proto_ids[bs, i]]
                    miss_emb_prob = miss_emb_prob / miss_emb_prob.sum()

                    # check non-zero
                    # fix nan
                    miss_emb_prob[torch.isnan(miss_emb_prob)] = 0
                    if miss_emb_prob.sum() == 0:
                        miss_emb_prob = torch.ones(len(global_prototype[0])) / len(global_prototype[0]) # quick fix

                    miss_emb_ids = np.random.choice(list(range(len(global_prototype[0]))), img_seq_len, p=miss_emb_prob.cpu().numpy())

                    miss_emb = []
                    for idx, img_proto_id in enumerate(miss_emb_ids):
                        img_proto_ids.append(img_proto_id)
                        img_proto = img_protos[img_proto_id]
                        img_proto_ = torch.tensor(img_proto[0]).to(img_embs.device) + torch.normal(0, img_proto[1], img_proto[0].shape).to(img_embs.device)

                        # logger.debug(f"img_proto_.shape: {img_proto_.shape}")
                        # miss_emb.append(img_proto_ - img_feats[bs, idx, :])
                        miss_emb.append(img_proto_)

                    miss_emb = torch.stack(miss_emb)
                    # miss_emb = np.array(miss_emb)

                    # logger.debug(f"miss_emb.shape: {miss_emb.shape}")

                    # FIXME: check
                    # if self.stage == 2:
                    #     zero_inp = torch.zeros_like(image[0]).to(image[0].device)
                    #     _, _, zero_feats = self.encode_img([zero_inp], return_feats=True)
                    #     miss_emb += zero_feats

                    # logger.debug(f"miss_emb.shape: {miss_emb.shape}")

                    # miss_embs.append(miss_emb)
                    miss_embs.append(miss_emb)

                miss_embs = torch.stack(miss_embs).to(image[0].device).to(torch.float16)
                # miss_embs = np.array(miss_embs).astype(np.float16)

                img_proto_ids = np.array(img_proto_ids).reshape(batch_size, img_seq_len)

                # logger.debug(f"miss_embs.shape: {miss_embs.shape}")

                miss_embs = miss_embs.reshape(img_embs.shape[0], -1, img_feats.shape[-1]).to(image[0].device)
                img_embs = self.proj(torch.stack([img_feats, miss_embs]).mean(0))

                proto_embs = []

                for bs in range(batch_size):
                    proto_emb = []
                    for i in range(img_seq_len):
                        proto_emb.append(self.proj(torch.tensor(img_protos[img_proto_ids[bs, i]][0]).to(image[0].device).to(torch.float16)))
                    proto_embs.append(torch.stack(proto_emb))
                
                proto_embs = torch.stack(proto_embs)

                sim_loss += get_sim_loss(img_embs, proto_embs, img_proto_ids, img_protos, txt_embs, txt_proto_ids, txt_protos)

        else:
            img_embs, img_atts = self.encode_img(image)

        # logger.debug(f"img_embs.shape: {img_embs.shape}")
        # logger.debug(f"img_atts.shape: {img_atts.shape}")

        img_embs = self.layer_norm(img_embs)

        img_embs, img_atts = self.prompt_wrap(img_embs, img_atts)

        self.tokenizer.padding_side = "right"
        text = [t + self.eos_token for t in samples["input_text"]]

        to_regress_tokens = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.cfg.model.max_length, add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == 0, -100)

        empty_targets = torch.ones([img_atts.shape[0], img_atts.shape[1] + 1], dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embs.shape[0]
        bos = (
            torch.ones([batch_size, 1], dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device)
            * self.tokenizer.bos_token_id
        )
        bos_embeds = self.embed_tokens(bos)
        bos_atts = img_atts[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        to_regress_token_att = to_regress_tokens.attention_mask

        # logger.debug(f"to_regress_tokens.input_ids.shape: {to_regress_tokens.input_ids.shape}")
        # logger.debug(f"img_embs.shape: {img_embs.shape}")
        # logger.debug(f"to_regress_embeds.shape: {to_regress_embeds.shape}")
        # logger.debug(f"to_regress_token_att.shape: {to_regress_token_att.shape}")

        if self.cfg.train.modal_hetero and mm_str == "mean" and mm_mean_type == "original" and client_type == 3:
            # logger.debug(f"Use strategy {mm_str}, mean_type {mm_mean_type}, client type {client_type}, processing text")
            means = self.cfg.payload["means"]

            to_regress_tokens = torch.from_numpy(means[2]).unsqueeze(0).expand(img_embs.shape[0], -1).to(img_embs.device)
            targets = to_regress_tokens.masked_fill(to_regress_tokens == 0, -100)

            empty_targets = torch.ones([img_atts.shape[0], img_atts.shape[1] + 1], dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embs.shape[0]
            bos = (
                torch.ones([batch_size, 1], dtype=to_regress_tokens.dtype, device=to_regress_tokens.device)
                * self.tokenizer.bos_token_id
            )
            bos_embeds = self.embed_tokens(bos)
            bos_atts = img_atts[:, :1]

            to_regress_embeds = self.embed_tokens(to_regress_tokens)

            to_regress_token_att = to_regress_tokens != 0
            to_regress_token_att = to_regress_token_att.to(img_embs.device)

            # logger.debug(f"to_regress_tokens.shape: {to_regress_tokens.shape}")
            # logger.debug(f"img_embs.shape: {img_embs.shape}")
            # logger.debug(f"to_regress_embeds.shape: {to_regress_embeds.shape}")
            # logger.debug(f"to_regress_token_att.shape: {to_regress_token_att.shape}")

        elif self.cfg.train.modal_hetero and mm_str == "mean" and mm_mean_type == "token" and client_type == 3:
            # logger.debug(f"Use strategy {mm_str}, mean_type {mm_mean_type}, client type {client_type}, processing text")
            means = self.cfg.payload["means"]
            to_regress_embeds = torch.from_numpy(means[2]).unsqueeze(0).expand(img_embs.shape[0], -1, -1).to(img_embs.device)

            # logger.debug(f"to_regress_embeds.shape: {to_regress_embeds.shape}")
        elif self.cfg.train.modal_hetero and mm_str == "prototype" and client_type == 3:
            # logger.debug(f"Use strategy {mm_str}, computing prototype loss")
            global_prototype = getattr(self.cfg.payload, "global_prototype", None)
            # TODO
            if global_prototype is None:
                # logger.debug("no global prototype")
                means = self.cfg.payload["means"]
                to_regress_embeds = torch.from_numpy(means[2]).unsqueeze(0).expand(img_embs.shape[0], -1, -1).to(img_embs.device)
            else:
                def get_img_emb():
                    img_embs, img_atts, img_feats = self.encode_img(image, return_feats=True)
                    return img_feats

                img_feats = get_img_emb()

                img_protos = global_prototype[0]
                txt_protos = global_prototype[1]
                img_proto_props, img_proto_ids = get_img_proto_props(img_feats, global_prototype[0])
                cross_matrix = global_prototype[2]
                txt_proto_weight = np.array([txt_proto[2] for txt_proto in txt_protos])

                miss_loss += get_miss_loss(img_feats, img_protos)

                # logger.debug(f"txt_protos: {txt_protos}")
                miss_embs = []

                batch_size, img_seq_len, img_embed_dim = img_feats.shape
                txt_seq_len = self.cfg.model.max_length

                img_proto_props = img_proto_props.reshape(batch_size, img_seq_len, -1)
                img_proto_ids = img_proto_ids.reshape(batch_size, -1)

                txt_proto_ids = []

                for bs in range(batch_size):
                    miss_emb_prob = torch.zeros(len(txt_protos))

                    for i in range(img_seq_len):
                        miss_emb_prob += cross_matrix[:, img_proto_ids[bs, i]]
                    miss_emb_prob = miss_emb_prob / miss_emb_prob.sum()

                    # check non-zero
                    # fix nan
                    miss_emb_prob[torch.isnan(miss_emb_prob)] = 0
                    if miss_emb_prob.sum() == 0:
                        miss_emb_prob = torch.ones(len(global_prototype[0])) / len(global_prototype[0])  # quick fix

                    miss_emb_ids = np.random.choice(list(range(len(txt_protos))), txt_seq_len, p=miss_emb_prob.cpu().numpy())

                    miss_emb = []
                    for txt_proto_id in miss_emb_ids:
                        txt_proto_ids.append(txt_proto_id)
                        txt_proto = txt_protos[txt_proto_id]
                        txt_proto_ = torch.tensor(txt_proto[0]).to(img_feats.device) + torch.normal(0, txt_proto[1], txt_proto[0].shape).to(img_feats.device)
                        miss_emb.append(txt_proto_)

                    miss_emb = torch.stack(miss_emb)

                    # logger.debug(f"miss_emb.shape: {miss_emb.shape}")
                    miss_embs.append(miss_emb)

                txt_proto_ids = np.array(txt_proto_ids).reshape(batch_size, txt_seq_len)

                # convert to float16
                miss_embs = torch.stack(miss_embs).to(image[0].device).to(torch.float16)

                # sim_loss += get_sim_loss(img_feats, img_proto_ids, img_protos, miss_embs, txt_proto_ids, txt_protos)

                # miss_embs = np.array(miss_embs).astype(np.float16)
                # logger.debug(f"miss_embs.shape: {miss_embs.shape}")
                to_regress_embeds = miss_embs.reshape(img_feats.shape[0], txt_seq_len, -1).to(image[0].device)

        inputs_embeds = torch.cat([bos_embeds, img_embs, to_regress_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, img_atts, to_regress_token_att], dim=1)

        # TODO: prototype loss
        if self.cfg.train.modal_hetero and mm_str == "prototype":
            global_prototype = getattr(self.cfg.payload, "global_prototype", None)

            if global_prototype is None:
                # logger.debug("no global prototype")
                pass
            else:
                # logger.debug("use strategy prototype, computing prototype loss")
                pass

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss + self.cfg.train.loss.weight_proto * miss_loss + self.cfg.train.loss.weight_sim * sim_loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad}
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {"model": state_dict, "config": self.cfg, "epoch": current_epoch, "step": global_step}
        os.makedirs(os.path.join(self.cfg.save_dir, "checkpoints"), exist_ok=True)
        if getattr(self.cfg, "client_idx", None) is not None:
            client_idx = self.cfg.client_idx
            round_idx = self.cfg.round_idx
            save_to = os.path.join(
                self.cfg.save_dir,
                "checkpoints",
                "checkpoint_client{}_round{}_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(client_idx, round_idx, current_epoch, global_step, eval_res["Bleu_4"], eval_res["CIDEr"]),
            )
            logger.info("Saving checkpoint for client {} at round {} at step {} to {}.".format(client_idx, round_idx, global_step, save_to))
        else:
            save_to = os.path.join(
                self.cfg.save_dir,
                "checkpoints",
                "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res["Bleu_4"], eval_res["CIDEr"]),
            )
        logger.info("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        to_regress_tokens = self.tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=False,
        )

        image = samples["image"]
        img_embs, img_atts = self.encode_img(image)
        img_embs = self.layer_norm(img_embs)
        img_embs, img_atts = self.prompt_wrap(img_embs, img_atts)

        batch_size = img_embs.shape[0]
        bos = torch.ones([batch_size, 1], dtype=img_atts.dtype, device=img_atts.device) * self.tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = img_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embs], dim=1)
        attention_mask = torch.cat([bos_atts, img_atts], dim=1)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.cfg.model.num_beams,
            do_sample=self.cfg.model.do_sample,
            min_new_tokens=self.cfg.model.min_new_tokens,
            max_new_tokens=self.cfg.model.max_new_tokens,
            repetition_penalty=self.cfg.model.repetition_penalty,
            length_penalty=self.cfg.model.length_penalty,
            # temperature=self.cfg.model.temperature,
            # top_p=self.cfg.model.top_p,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens["input_ids"]]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("</s>")[0].strip()
        output_text = output_text.replace("<unk>", "")
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i["ref"])
            hypo.extend(i["hypo"])
            ids.extend(i["id"])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.cfg.save_dir, "result")
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + ".json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, "refs.json"), "w"))

        self.print(eval_res)
        logger.info(eval_res)

        val_score = 0
        for score_type, weight in zip(self.cfg.train.monitor_metrics, self.cfg.train.monitor_weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        to_regress_tokens = self.tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=False,
        )

        image = samples["image"]
        img_embs, img_atts = self.encode_img(image)
        img_embs = self.layer_norm(img_embs)
        img_embs, img_atts = self.prompt_wrap(img_embs, img_atts)

        batch_size = img_embs.shape[0]
        bos = torch.ones([batch_size, 1], dtype=img_atts.dtype, device=img_atts.device) * self.tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = img_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embs], dim=1)
        attention_mask = torch.cat([bos_atts, img_atts], dim=1)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.cfg.model.num_beams,
            do_sample=self.cfg.model.do_sample,
            min_new_tokens=self.cfg.model.min_new_tokens,
            max_new_tokens=self.cfg.model.max_new_tokens,
            repetition_penalty=self.cfg.model.repetition_penalty,
            length_penalty=self.cfg.model.length_penalty,
            # temperature=self.cfg.model.temperature,
            # top_p=self.cfg.model.top_p,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens["input_ids"]]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i["ref"])
            hypo.extend(i["hypo"])
            ids.extend(i["id"])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.cfg.save_dir, "result")
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, "test_refs.json"), "w"))

        self.print(f"Test result of {self.cfg.model.delta_file}: {eval_res}")
        logger.info(f"Test result of {self.cfg.model.delta_file}: {eval_res}")

    def configure_optimizers(self):
        if self.stage == None or self.stage == 1:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.train.num_epochs, eta_min=1e-6)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.visual_encoder.parameters(), "lr": self.cfg.train.lr},
                    {"params": self.proj.parameters(), "lr": self.cfg.train.lr_proj},
                    {"params": self.layer_norm.parameters(), "lr": self.cfg.train.lr_proj},
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.train.num_epochs, eta_min=1e-6)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    def get_trainable_params(self):
        params = {}

        for name, param in self.named_parameters():
            if param.requires_grad:
                params[name] = param

        return params

    def set_trainable_params(self, params: torch.Dict[str, torch.Tensor]) -> None:
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name]

    def get_prototype(self, dm, return_all=False):
        # TODO
        client_type = getattr(self.cfg, "client_type", None)
        mm_str = getattr(self.cfg.train, "mm_strategy", None)
        mm_mean_type = getattr(self.cfg.train, "mm_mean_type", None)

        self.eval()
        self.to("cuda")

        # logger.debug(f"Client type: {client_type}, mm_str: {mm_str}, mm_mean_type: {mm_mean_type}")

        cross_protos = []

        with torch.no_grad():
            for samples in dm.train_dataloader():
                image = samples["image"]
                image = [img.to(self.device) for img in image]
                # logger.debug(f"self.visual_encoder.device: {self.visual_encoder.device}")
                # logger.debug(f"self.proj.device: {self.proj.device}")
                # logger.debug(f"self.layer_norm.device: {self.layer_norm.device}")
                # logger.debug(f"self.embed_tokens.device: {self.embed_tokens.device}")

                # logger.debug(f"image[0].device: {image[0].device}")
                # logger.debug(f"image[1].device: {image[1].device}")
                # img_embs1, _ = self.encode_img([image[0]])
                # img_embs2, _ = self.encode_img([image[1]])
                _, _, img_embs1 = self.encode_img([image[0]], return_feats=True)
                _, _, img_embs2 = self.encode_img([image[1]], return_feats=True)

                # img_embs, img_atts = self.encode_img(image)
                # img_embs1 = self.layer_norm(img_embs1)
                # img_embs2 = self.layer_norm(img_embs2)

                # logger.debug(f"img_embs1.shape: {img_embs1.shape}")
                # logger.debug(f"img_embs2.shape: {img_embs2.shape}")

                text = [t + self.eos_token for t in samples["input_text"]]

                text_token = self.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.cfg.model.max_length, add_special_tokens=False
                ).input_ids.to(image[0].device)
                text_embs = self.embed_tokens(text_token)
                # logger.debug(f"text_embs.shape: {text_embs.shape}")

                cross_proto = [img_embs1.cpu().numpy(), img_embs2.cpu().numpy(), text_embs.cpu().numpy()]
                cross_protos.append(cross_proto)

        # cluster cross_prototypes
        img_protos = []
        text_protos = []

        for cross_proto in cross_protos:
            # logger.debug(f"cross_proto[0].shape: {cross_proto[0].shape}")
            # logger.debug(f"cross_proto[1].shape: {cross_proto[1].shape}")
            # logger.debug(f"cross_proto[2].shape: {cross_proto[2].shape}")
            img_proto = np.mean(np.array([cross_proto[0], cross_proto[1]]), axis=2)  # 4,49,1024 + 4,49,1024 = 4,49,1024
            text_proto = cross_proto[2] # 4,60,4096

            img_protos.append(img_proto)
            text_protos.append(text_proto)

        img_protos = np.concatenate(img_protos, axis=0) # n,49,2048
        text_protos = np.concatenate(text_protos, axis=0) # n,60,4096

        img_seq_len = img_protos.shape[1]
        txt_seq_len = text_protos.shape[1]

        # logger.debug(f"img_protos.shape: {img_protos.shape}")
        # logger.debug(f"text_protos.shape: {text_protos.shape}")

        # n,49,1024 -> 49*n,1024
        img_protos = img_protos.reshape(-1, img_protos.shape[-1])
        text_protos = text_protos.reshape(-1, text_protos.shape[-1])

        # logger.debug(f"img_protos.shape: {img_protos.shape}")
        # logger.debug(f"text_protos.shape: {text_protos.shape}")

        from sklearn.cluster import KMeans

        img_cluster = KMeans(n_clusters=self.cfg.train.cross_fedproto.k, random_state=self.cfg.seed).fit(img_protos)
        text_cluster = KMeans(n_clusters=self.cfg.train.cross_fedproto.k, random_state=self.cfg.seed).fit(text_protos)

        img_prototypes = img_cluster.cluster_centers_
        text_prototypes = text_cluster.cluster_centers_

        # logger.debug(f"img_prototypes.shape: {img_prototypes.shape}")
        # logger.debug(f"text_prototypes.shape: {text_prototypes.shape}")

        # get cross prototypes matrix
        cross_matrix = np.zeros((self.cfg.train.cross_fedproto.k, self.cfg.train.cross_fedproto.k))
        proto_cnt_i = np.zeros((self.cfg.train.cross_fedproto.k,))
        proto_cnt_t = np.zeros((self.cfg.train.cross_fedproto.k,))
        proto_std_i = np.zeros((self.cfg.train.cross_fedproto.k,))
        proto_std_t = np.zeros((self.cfg.train.cross_fedproto.k,))

        img_proto_labels = img_cluster.labels_
        text_proto_labels = text_cluster.labels_

        for i in range(self.cfg.train.cross_fedproto.k):
            proto_cnt_i[i] = np.sum(img_proto_labels == i)
            proto_cnt_t[i] = np.sum(text_proto_labels == i)
            proto_std_i[i] = np.std(img_protos[img_proto_labels == i])
            proto_std_t[i] = np.std(text_protos[text_proto_labels == i])

        for k in range(len(dm.train_dataloader().dataset)):
            for i in range(self.cfg.train.cross_fedproto.k):
                for j in range(self.cfg.train.cross_fedproto.k):
                    cross_matrix[i][j] += min(
                        np.sum(img_proto_labels[k * img_seq_len : (k + 1) * img_seq_len] == i),
                        np.sum(text_proto_labels[k * txt_seq_len : (k + 1) * txt_seq_len] == j),
                    )  # FIXME: hard code seq len

        # cross_matrix = cross_matrix / cross_matrix.sum()

        # logger.debug(f"cross_matrix.shape: {cross_matrix.shape}")
        # logger.debug(f"cross_matrix: {cross_matrix}")

        # logger.debug(f"proto_cnt_i: {proto_cnt_i}")
        # logger.debug(f"proto_cnt_t: {proto_cnt_t}")
        # logger.debug(f"proto_std_i: {proto_std_i}")
        # logger.debug(f"proto_std_t: {proto_std_t}")

        self.to("cpu")

        if return_all:
            return (img_prototypes, text_prototypes, cross_matrix, proto_cnt_i, proto_cnt_t, proto_std_i, proto_std_t, img_protos, text_protos, img_cluster, text_cluster)

        return (img_prototypes, text_prototypes, cross_matrix, proto_cnt_i, proto_cnt_t, proto_std_i, proto_std_t)
