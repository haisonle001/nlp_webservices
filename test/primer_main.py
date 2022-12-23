# coding: utf8
from pytorch_lightning.accelerators import accelerator
import torch
import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AutoTokenizer,
    LEDTokenizer,
    LEDConfig,
    LEDForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers import Adafactor
from longformer.sliding_chunks import pad_to_window_size
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig
import pandas as pd
import pdb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from datasets import load_dataset, load_metric
from dataloader import (
    get_dataloader_summ,
    get_dataloader_pretrain,
    get_dataloader_summiter,
)
import json
from pathlib import Path
# from rouge import Rouge
import py_vncorenlp

# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="C:/Work/NLP/PRIMER")
def initialize_args():
        # parse = argparse.ArgumentParser()
    ########################
    # General
        parser = argparse.ArgumentParser()
    ########################
    # General
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use")
        parser.add_argument(
                "--accelerator", default='gpu', type=str, help="Type of accelerator"
        )
        parser.add_argument(
                "--mode", default="predict", choices=["pretrain", "train", "test","predict"]
        )
        parser.add_argument(
                "--debug_mode", action="store_true", help="set true if to debug"
        )
        parser.add_argument(
                "--compute_rouge",
                action="store_true",
                help="whether to compute rouge in validation steps",
                default=False,
        )
        parser.add_argument(
                "--saveRouge",
                action="store_true",
                help="whether to compute rouge in validation steps",
        )

        parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
        ####
        parser.add_argument(
                "--model_path", type=str, default="."
        )
        parser.add_argument("--ckpt_path", type=str, default=None)
        parser.add_argument("--saveTopK", default=3, type=int)
        parser.add_argument(
                "--resume_ckpt",
                type=str,
                help="Path of a checkpoint to resume from",
                default="C:/Work/NLP/PRIMER/step70-vloss1.82-avgr0.5055.ckpt",
        )

        ####
        parser.add_argument("--data_path", type=str, default=".")
        parser.add_argument("--dataset_name", type=str, default="wcep")
        parser.add_argument("--tokenizer", type=str, default="facebook/bart-base")
        parser.add_argument(
                "--num_workers",
                type=int,
                default=0,
                help="Number of workers to use for dataloader",
        )

        parser.add_argument("--batch_size", default=1, type=int)
        parser.add_argument("--max_length_input", default=4096, type=int)
        parser.add_argument("--max_length_tgt", default=1024, type=int)
        parser.add_argument("--min_length_tgt", default=0, type=int)
        parser.add_argument("--join_method", type=str, default="concat_start_wdoc_global")
        parser.add_argument(
                "--attention_dropout", type=float, default=0.1, help="attention dropout"
        )
        parser.add_argument(
                "--attention_mode",
                type=str,
                default="sliding_chunks",
                help="Longformer attention mode",
        )
        parser.add_argument(
                "--attention_window", type=int, default=512, help="Attention window"
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
                "--adafactor", action="store_true", help="Use adafactor optimizer"
        )
        parser.add_argument(
                "--fp32",
                action="store_true",
                help="default is fp16. Use --fp32 to switch to fp32",
        )
        parser.add_argument(
                "--grad_ckpt",
                action="store_true",
                help="Enable gradient checkpointing to save memory",
        )
        parser.add_argument(
                "--rand_seed",
                type=int,
                default=0,
                help="seed for random sampling, useful for few shot learning",
        )
                ########################
        # For training
        ####
        parser.add_argument(
                "--primer_path",
                type=str,
                default="C:/Work/NLP/PRIMERA-github/text-summarization/PRIMER/PRIMER_model",
        )
        parser.add_argument(
                "--limit_valid_batches",
                type=int,
                default=None,
        )
        parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
        parser.add_argument(
                "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
        )
        parser.add_argument(
                "--report_steps", type=int, default=50, help="Number of report steps"
        )
        parser.add_argument(
                "--val_check_interval",
                default=1.0,
                type=float,
                help="Number of steps to evaluate",
        )
        parser.add_argument(
                "--accum_data_per_step", type=int, default=16, help="Number of data per step"
        )
        parser.add_argument(
                "--total_steps", type=int, default=50000, help="Number of steps to train"
        )
        parser.add_argument(
                "--num_train_data",
                type=int,
                default=-1,
                help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
        )
        parser.add_argument(
                "--remove_masks",
                action="store_true",
                help="remove all the masks in pretraining",
        )
        parser.add_argument(
                "--fix_lr",
                action="store_true",
                help="use fix learning rate",
        )
        parser.add_argument(
                "--test_imediate",
                action="store_true",
                help="test on the best checkpoint",
        )
        parser.add_argument(
                "--fewshot",
                action="store_true",
                help="whether this is a run for few shot learning",
        )
        parser.add_argument(
                "--eval_steps",
                type=int,
                default=5000,
                help="Number of steps to evaluate in the pre-training stage.",
        )
        ########################
        # For testing
        parser.add_argument(
                "--limit_test_batches",
                type=int,
                default=None,
                help="Number of batches to test in the test mode.",
        )
        parser.add_argument("--beam_size", type=int, default=4, help="size of beam search")
        parser.add_argument(
                "--length_penalty",
                type=float,
                default=1.0,
                help="length penalty, <1 encourage shorter message and >1 encourage longer messages",
        )

        parser.add_argument(
                "--mask_num",
                type=int,
                default=0,
                help="Number of masks in the input of summarization data",
        )
        parser.add_argument(
                "--test_batch_size",
                type=int,
                default=-1,
                help="batch size for test, used in few shot evaluation.",
        )
        parser.add_argument(
                "--applyTriblck",
                action="store_true",
                help="whether apply trigram block in the evaluation phase",
        )

        arg =parser.parse_args()  # Get pad token id
        ####################

        ####################

        arg.acc_batch = arg.accum_data_per_step // arg.batch_size
        arg.data_path = os.path.join(arg.data_path, arg.dataset_name)
        return arg

args=initialize_args()
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class PRIMERSummarizerLN(pl.LightningModule):
    def __init__(self, args):
        super(PRIMERSummarizerLN, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        self.pad_token_id = self.tokenizer.pad_token_id
        config = LongformerEncoderDecoderConfig.from_pretrained(args.primer_path)
        config.gradient_checkpointing = args.grad_ckpt
        self.model = LEDForConditionalGeneration(config)

        self.use_ddp = args.accelerator == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        if args.mode=='pretrain' or args.mode=='test' or args.mode =='train' or args.mode=='predict':
            # The special token is added after each document in the pre-processing step.
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<doc-sep>"]}
            )
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )
        attention_mask[input_ids == self.pad_token_id] = 0
        if isinstance(self.model, LongformerEncoderDecoderForConditionalGeneration):
            # global attention on one token for all model params to be used,
            # which is important for gradient checkpointing to work
            attention_mask[:, 0] = 2

            if self.args.join_method == "concat_start_wdoc_global":
                attention_mask[input_ids == self.docsep_token_id] = 2

            if self.args.attention_mode == "sliding_chunks":
                half_padding_mod = self.model.config.attention_window[0]
            elif self.args.attention_mode == "sliding_chunks_no_overlap":
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError

            input_ids, attention_mask = pad_to_window_size(
                # ideally, should be moved inside the LongformerModel
                input_ids,
                attention_mask,
                half_padding_mod,
                self.pad_token_id,
            )
        # print(attention_mask.size(),input_ids.size()) 
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):

        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = decoder_input_ids != self.pad_token_id
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        lm_logits = outputs[0]
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        return lm_logits

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, output_ids):
        lm_logits = self.forward(input_ids, output_ids)
        labels = output_ids[:, 1:].clone()
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, output_ids = batch
        loss = self.shared_step(input_ids, output_ids)
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size": input_ids.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        return loss

    def compute_rouge_batch(self, input_ids, output_ids, gold_str):
        scorer = load_metric("rouge")
        # rouge = Rouge()
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_return_sequences=4,
            num_beam_groups=4,
            diversity_penalty=1.0,
            num_beams=4,
            length_penalty=2.0,
            # length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
            # no_repeat_ngram_size=3 if self.args.applyTriblck else None,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        if self.args.mode == "test":
            if self.args.applyTriblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_%d_%d_triblck_beam=%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                        self.args.beam_size,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_%d_%d_beam=%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                        self.args.beam_size,
                    ),
                )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # idx = len(os.listdir(output_dir))
            idx=1
        result_batch = []
        for ref, pred in zip(gold_str, generated_str):
            if self.args.mode == "test":
                with open(os.path.join(output_dir, "%d.txt" % (idx)), "a") as of:
                    of.write(pred)
                # idx += 1

            s = scorer.compute(
                predictions=[pred],
                references=[ref],
                # use_agregator=False,
                use_stemmer=False,
            )
            result_batch.append(
                (
                    s["rouge1"][0].recall,
                    s["rouge1"][0].precision,
                    s["rouge1"][0].fmeasure,
                    s["rouge2"][0].recall,
                    s["rouge2"][0].precision,
                    s["rouge2"][0].fmeasure,
                    s["rougeL"][0].recall,
                    s["rougeL"][0].precision,
                    s["rougeL"][0].fmeasure,
                    s["rougeLsum"][0].recall,
                    s["rougeLsum"][0].precision,
                    s["rougeLsum"][0].fmeasure,
                )
            )

            #another way to calculate rouge
            # s = rouge.get_scores(pred,ref,avg=False)
            # result_batch.append(
            #     (
            #         s[0]["rouge-1"]['r'],
            #         s[0]["rouge-1"]['p'],
            #         s[0]["rouge-1"]['f'],
            #         s[0]["rouge-2"]['r'],
            #         s[0]["rouge-2"]['p'],
            #         s[0]["rouge-2"]['f'],
            #         s[0]["rouge-l"]['r'],
            #         s[0]["rouge-l"]['p'],
            #         s[0]["rouge-l"]['f'],
            #         s[0]["rouge-2"]['r'],
            #         s[0]["rouge-2"]['p'],
            #         s[0]["rouge-2"]['f'],
            #     )
            # )
        return result_batch

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        if self.args.mode=='pretrain':
            input_ids, output_ids = batch
        else:
            input_ids, output_ids, tgt = batch
        loss = self.shared_step(input_ids, output_ids)
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids, output_ids, tgt)
            return {"vloss": loss, "rouge_result": result_batch}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg
        if output_file:
            csv_name = (
                args.model_path
                + output_file
                + "_beam=%d" % (self.args.beam_size)
                + "_lenPenalty=%.2f" % (self.args.length_penalty)
                + "_triblck=%s" % (self.args.applyTriblck)
                + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
            rouge_results.to_csv(csv_name)

        avgr = (avg[2] + avg[5] + avg[8]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, \
            Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        return names, metrics, avgr

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_rouge:
            names, metrics, avgr = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgr", avgr)
            return {
                "avg_val_loss": vloss,
                "avgr": avgr,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}
    
    def generate_predict(self, input_ids, gold_str):
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=3 ,
        )
        # generated_ids = self.model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     use_cache=True,
        #     max_length=args.max_length_tgt,
        #     min_length=args.min_length_tgt,
        #     num_return_sequences=4,
        #     num_beam_groups=4,
        #     diversity_penalty=1.0,
        #     num_beams=4,
        #     # length_penalty=2.0,
        #     length_penalty=args.length_penalty,
        #     no_repeat_ngram_size=3,
        #     early_stopping=True,
        #     # no_repeat_ngram_size=3 if args.applyTriblck else None,
        # )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        print(generated_str)

        output_dir = os.path.join(
            self.args.model_path,
            "predicted_folder"    
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        idx = len(os.listdir(output_dir))

        for pred in generated_str:
            with open(os.path.join(output_dir, "prediction.txt"), "w", encoding="utf-8") as of:
                of.write(pred)
            with open(os.path.join(output_dir, "prediction.jsonl_5" ), "w", encoding="utf-8") as fo:
                json.dump(pred, fo, ensure_ascii=False, indent=4)
            idx += 1
        return []

    def prediction_step(self, batch, batch_idx):
        input_ids, output_ids, tgt = batch
        generate = self.generate_predict(input_ids, tgt)
        return True

    def test_step(self, batch, batch_idx):
        if self.args.mode=='predict':
            return self.prediction_step(batch, batch_idx)
        else:
            return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        if self.args.mode!='predict':
            tloss = torch.stack([x["vloss"] for x in outputs]).mean()
            self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
            output_file = "test_%s_%d_%d" % (
                self.args.dataset_name,
                self.args.max_length_input,
                self.args.max_length_tgt,
            )
            output_file = (
                output_file
                + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
                if self.args.fewshot
                else output_file
            )
            output_file = (
                output_file + "_triblck" if self.args.applyTriblck else output_file
            )

            names, metrics, avgr = self.compute_rouge_all(outputs, output_file=output_file)
            metrics = [tloss, avgr] + metrics
            names = ["tloss", "avgr"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgr", avgr)
            return {"avg_test_loss": tloss, "avgr": avgr, "log": logs, "progress_bar": logs}
        else:
            return 0


def pretrain(args):
    args.compute_rouge = False
    model = PRIMERSummarizerLN(args)
    # if args.resume_ckpt:
    #     model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
    # else:
    #     model = PRIMERSummarizerLN(args)


    # initialize checkpoints
    if args.ckpt_path is None:
        args.ckpt_path = args.model_path + "pretrain_checkpoints/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}",
        save_top_k=args.saveTopK,
        monitor="vloss",
        mode="min",
    )

    # initialize loggers
    logger = TensorBoardLogger(args.model_path + "tb_logs", name="pretrain_process")

    # initialize trainers
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.acc_batch,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.acc_batch,
        val_check_interval=args.eval_steps * args.acc_batch,
        # plugins=DDPPlugin(find_unused_parameters=False)
        plugins=DDPPlugin(find_unused_parameters=True)
        if args.accelerator == "ddp"
        else None,
        logger=logger,
        log_every_n_steps=5,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate * args.acc_batch
        if not args.debug_mode
        else 1,
        precision=32 if args.fp32 else 16,
        resume_from_checkpoint=args.resume_ckpt,
        accelerator=args.accelerator,
    )

    # load datasets
    train_dataloader = get_dataloader_pretrain(
        args,
        args.data_path,
        "train",
        args.num_workers,
        use_ddp=(args.gpus > 1),
        mask_id=model.tokenizer.mask_token_id,
    )
    valid_dataloader = get_dataloader_pretrain(
        args,
        args.data_path,
        "valid",
        args.num_workers,
        use_ddp=(args.gpus > 1),
        mask_id=model.tokenizer.mask_token_id,
    )

    trainer.fit(model, train_dataloader, valid_dataloader)


def train(args):
    args.compute_rouge = True

    if args.resume_ckpt:
        model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = PRIMERSummarizerLN(args)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = args.model_path + "summ_checkpoints/"
    if args.num_train_data != -1:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.ckpt_path,
            filename="{step}-{vloss:.2f}-{avgr:.4f}",
            save_top_k=args.saveTopK,
            monitor="avgr",
            mode="max",
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.ckpt_path,
            filename="{step}-{vloss:.2f}-{avgr:.4f}",
            save_top_k=args.saveTopK,
            monitor="avgr",
            mode="max",
        )

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name="my_model")

    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        # max_steps=args.total_steps,
        max_steps=150000,
        replace_sampler_ddp=True,
        accumulate_grad_batches=args.acc_batch,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=1
        if args.num_train_data > 100 or args.num_train_data == -1
        else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate * args.acc_batch
        if not args.debug_mode
        else 1,
        precision=32 if args.fp32 else 16,
        plugins=DDPPlugin(find_unused_parameters=True)
        if args.accelerator == "ddp"
        else None,
        accelerator=args.accelerator,
    )

    # load datasets
    if os.path.isdir(args.data_path):
        dataset = torch.load(args.data_path + "train.pt")
    train_dataloader = get_dataloader_summ(
        args, dataset, model.tokenizer, "train", args.num_workers, True
    )
    if os.path.exists(args.data_path + "val.pt"):
        dataset = torch.load(args.data_path + "val.pt")
    else:
         dataset = torch.load(args.data_path + "valid.pt")
    valid_dataloader = get_dataloader_summ(
        args, dataset, model.tokenizer, "validation", args.num_workers, True
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    args.compute_rouge = True
    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.acc_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        precision=32 if args.fp32 else 16,
        accelerator=args.accelerator,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
    )
    # load from checkpoints
    if args.resume_ckpt:
        model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = PRIMERSummarizerLN(args)

    # load dataset
    if os.path.isdir(args.data_path):
        dataset = torch.load(args.data_path + "test.pt")
    else:
        dataset = torch.load(args.data_path)
    test_dataloader = get_dataloader_summ(
        args, dataset, model.tokenizer, "test", args.num_workers, False
    )
    # test
    trainer.test(model, test_dataloader)

def predictt(args,model):
    if not args.resume_ckpt:
        print("Resume checkpoint is not provided.")
    else:
        docs='Với 349 phiếu ủng_hộ so với 238 phiếu chống , chính_phủ của Thủ_tướng Boris_Johnson ngày 18/7 ( giờ_địa_phương ) đã vượt qua cuộc bỏ_phiếu tín_nhiệm tự đề_xuất sau khi các thành_viên đảng Bảo_thủ cầm_quyền buộc phải lựa_chọn ủng_hộ chính_phủ nhằm tránh bầu_cử sớm . Trước đó , Công đảng đối_lập đã kêu_gọi thay_thế ông Boris_Johnson ngay_lập_tức bằng một thủ_tướng tạm_quyền cho đến khi người kế_nhiệm chính_thức được đảng Bảo_thủ bầu ra vào đầu tháng 9 . Công đảng đã tìm cách tổ_chức bỏ_phiếu tín_nhiệm đối_với cả chính_phủ và cá_nhân ông Boris_Johnson nhằm buộc ông phải rút_lui sớm hơn . Tuy_nhiên , Chính_phủ Anh phản_đối với lý_do như_vậy là không cần_thiết sau khi Thủ_tướng Boris_Johnson đã cam_kết sẽ rời bỏ chức_vụ khi có người thay_thế . Thay vào đó , đảng Bảo_thủ đề_xuất chỉ bỏ_phiếu tín_nhiệm đối_với chính_phủ . Ngày 7/7 vừa_qua , ông Boris_Johnson đã thông_báo từ_chức lãnh_đạo đảng Bảo_thủ , đồng_nghĩa với việc từ_chức Thủ_tướng Anh , trong bối_cảnh nhà_lãnh_đạo này vướng phải hàng_loạt bê_bối và đối_mặt với làn_sóng từ_chức của hàng chục thành_viên cấp cao trong chính_phủ . Tuy_nhiên , ông Boris_Johnson sẽ vẫn tiếp_tục vai_trò lãnh_đạo Chính_phủ Anh cho đến khi đảng Bảo_thủ bầu được lãnh_đạo mới . Các thành_viên trong đảng Bảo_thủ kỳ_vọng quá_trình bầu_cử nội_bộ 2 giai_đoạn sẽ hoàn_thành trước khi diễn ra hội_nghị thường_niên của đảng vào tháng 10 tới . Thông_báo từ_nhiệm của ông Boris_Johnson được đánh_giá là một quyết_định quan_trọng và trong trường_hợp này càng đáng chú_ý hơn . Trên thực_tế , ông Boris_Johnson không hề “ thất_cử ” . Tháng 12/2019 , ông đã lãnh_đạo Đảng Bảo_thủ giành chiến_thắng quyết_đoán trong tổng_tuyển_cử . Chính_phủ Bảo_thủ có được thế đa_số đáng_kể tại Hạ_viện và vài tuần trước ông đã vượt qua dễ_dàng cuộc bỏ_phiếu tín_nhiệm nội_bộ đảng . Tuy_nhiên , ông đang đi ngược_lại chiều gió bất_chấp việc có những món quà lớn nhất mà Hiến_pháp Anh có_thể “ ban_tặng ” cho một thủ_tướng - đa_số đủ để giúp thông_qua các dự_luật tại Hạ_viện và vị_thế để thúc_đẩy điều tương_tự ở Thượng_viện . Việc ông quyết_định từ_chức cho thấy nhiều điều trong cả quyền_lực và điểm yếu của vị_trí thủ_tướng . Nhìn bên ngoài , Văn_phòng Thủ_tướng Anh là nơi có sức_mạnh to_lớn . Về lý_thuyết , Thủ_tướng và đảng của họ chỉ tuân_thủ luật của các cuộc tổng_tuyển_cử . Tuy_nhiên , kể từ năm 1974 , các thủ_tướng có xu_hướng lên nắm quyền hoặc mất chức giữa các cuộc tổng_tuyển_cử , hoặc cả 2 với trường_hợp của bà Theresa_May và ông Boris_Johnson . Điều này là do ghế thủ_tướng được chọn_lựa phụ_thuộc vào sự tín_nhiệm của cả nội_các và đa_số nghị_sỹ Quốc_hội của chính_đảng nắm thế đa_số . Khi niềm tin không còn , các nhà_lãnh_đạo thường có xu_hướng từ_chức . Việc các thành_viên của đảng mất niềm tin có_thể thể_hiện qua nhiều cách ngoài lá phiếu tín_nhiệm , chẳng_hạn như lời kêu_gọi bầu_cử hoặc bằng cách từ_chức và từ_chối được bổ_nhiệm . Một thủ_tướng có_thể tìm cách phớt_lờ những tín_hiệu này , song tới cuối_cùng họ sẽ phải đối_mặt với các cuộc bỏ_phiếu bất_tín_nhiệm chính_thức của Quốc_hội hoặc bị Nữ_hoàng Elizabeth cách_chức . Có những đồn đoán rằng ông Johnson sẽ “ chiến_đấu ” sau hàng_loạt đơn_từ_chức bất_thường và những lá thư bất_tín_nhiệm trong vài ngày qua . Song thực_tế mọi chuyện đã khác . Trong cuộc bỏ_phiếu tín_nhiệm nội_bộ của đảng Bảo_thủ , ông nhận được 211 phiếu , hơn ngưỡng 180 cần_thiết để giành chiến_thắng . Tuy_nhiên , ông đã thua theo một nghĩa chính_trị rộng_lớn hơn , quan_trọng hơn , ông không thắng đủ để chấm_dứt những câu hỏi về năng_lực lãnh_đạo đã đeo_bám ông từ lâu . Ví_dụ , trong phần_lớn cuộc khủng_hoảng COVID-19 , cách tiếp_cận để giải_quyết vấn_đề của ông đa_phần hỗn_loạn và thiếu nhất_quán , phản_ánh một thực_tế là phong_cách của ông thiên về “ bức tranh lớn ” chứ không tập_trung vào chi_tiết , trong khi phong_cách sống lại không mấy phù_hợp với các đòi_hỏi của thời_đại dịch . Tuy_nhiên , những người ủng_hộ ông Boris_Johnson sẽ lập_luận rằng ông đã đạt được hai chiến_thắng quan_trọng . Thứ nhất là chiến_thắng trong cuộc tổng_tuyển_cử mang tính bước_ngoặt vào năm 2019 , khi Đảng Bảo_thủ ít_nhất cũng tạm_thời đã vẽ lại bản_đồ chính_trị của Anh bằng cách giành được một_số thành_trì lâu_đời của Công đảng , đặc_biệt là ở Midlands và Bắc_Anh . Thứ hai , ông có công lớn trong cuộc trưng_cầu_ý_dân về Brexit gây chia_rẽ vào năm 2016 và sau đó đưa Anh rời khỏi Liên_minh châu_Âu ( EU ) vào năm 2020 . Đây là lý_do tại_sao di_sản chính_trị của nhà_lãnh_đạo này có_thể sẽ là đề_tài gây tranh_cãi trong nhiều năm tới . Dù đã đạt được những thành_tựu lịch_sử , bao_gồm cả Brexit , quốc_gia này chắc_chắn sẽ bị chia_rẽ nhiều hơn là đoàn_kết bởi nhiệm_kỳ thủ_tướng gây tranh_cãi của ông Boris_Johnson . Ông Boris_Johnson có_thể đã từ_chức lãnh_đạo Đảng Bảo_thủ nhưng những rắc_rối của đảng cầm_quyền ở Anh còn lâu mới kết_thúc . Trong khi các nghị_sỹ Đảng Bảo_thủ phần_lớn đồng_tình rằng giờ là thời_điểm thích_hợp để Thủ_tướng Johnson ra đi , chính_đảng này đang rất chia_rẽ về việc người kế_nhiệm ông nên là ai . Nhiều khả_năng sẽ có những cuộc tranh_luận gay_gắt trong đảng những tuần tới , về cả việc liệu nước này có cần quay lại hướng tiếp_cận áp mức thuế thấp theo kiểu Thatcher , một_cách tiếp_cận nhà_nước quy_mô hẹp hơn sau khi lựa_chọn cách can_thiệp kiểu ông Boris_Johnson với các khoản chi_tiêu lớn , ảnh_hưởng bởi tình_hình khẩn_cấp về y_tế và kinh_tế trong giai_đoạn đại_dịch hay không . Sẽ có rất nhiều ứng_cử_viên cho vị_trí lãnh_đạo của đảng và nhân_vật sẽ ngồi vào ghế thủ_tướng . Các ứng_cử_viên đã được công_bố gồm Tổng_Chưởng lý Suella_Braverman và Chủ_tịch Uỷ_ban Đối_ngoại Tom_Tugendhat và dự_kiến những cái tên khác như Ngoại_trưởng Liz_Truss ; cựu Bộ_trưởng Giáo_dục Nadhim_Zahawi , người vừa được Johnson bổ_nhiệm làm Bộ_trưởng Tài_chính hôm 4/7 ; Bộ_trưởng Thương_mại Penny_Mordaunt , Bộ_trưởng Quốc_phòng Ben_Wallace và Bộ_trưởng Tư_pháp Dominic_Raab . Còn những ứng_cử_viên tiềm_năng khác như cựu Bộ_trưởng Y_tế Sajid_Javid và cựu Bộ_trưởng Tài_chính Rishi_Sunak , những người bất_ngờ từ_chức hôm 4/7 . Nhiệm_vụ quan_trọng đối_với việc tìm người thay_thế ông Boris_Johnson sẽ là chấm_dứt tình_trạng tê_liệt trong chính_phủ bằng một tầm nhìn rõ_ràng hậu Brexit trong quá_trình hồi_phục sau đại_dịch .<doc-sep>Theo thông_báo của đảng Bảo_thủ , cựu Bộ_trưởng Tài_chính Rishi_Sunak là người về nhất trong vòng bỏ_phiếu vừa_qua với 137 phiếu ủng_hộ . Trong khi đó , Ngoại_trưởng Liz_Truss đánh_bại Bộ_trưởng Thương_mại Penny_Mordaunt để giành tấm vé vào vòng cuối_cùng với tỉ_lệ phiếu 113 - 105 . Sau khi kết_quả được công_bố , cả hai ứng_viên đã chia_sẻ quan_điểm trên trang Twitter cá_nhân . " Xin gửi lời cảm_ơn tới các cộng_sự đã đặt niềm tin vào tôi . Tôi sẽ làm_việc ngày_đêm để truyền đi thông_điệp của chúng_ta tới thế_giới " , ông Sunak viết . Về phía Ngoại_trưởng Liz_Truss , bà nhấn_mạnh luôn sẵn_sàng và nỗ_lực để đạt được mục_tiêu từ ngày đầu của cuộc chạy_đua . Như_vậy , ông Rishi_Sunak và bà Liz_Truss sẽ đối_mặt trong vòng bỏ_phiếu quyết_định cuối_cùng của trên 170.000 đảng_viên đảng Bảo_thủ , dự_kiến tổ_chức từ cuối tháng 8 - 2/9 . Trước_mắt , hai ứng_viên này sẽ có phiên tranh_luận trực_tiếp trên truyền_hình vào ngày 25/7 và sẽ tiến_hành các chiến_dịch vận_động tranh_cử trong tháng 8 . Theo nhận_định của truyền_thông Anh , không_thể đoán trước kết_quả cuộc đua giữa ông Rishi_Sunak và bà Liz_Truss . Thực_tế cho thấy ông Sunak thường về nhất trong các vòng bỏ_phiếu vừa_qua , nhưng bà Liz_Truss được cho là có lợi_thế hơn ở vòng cuối bởi những người bỏ_phiếu trong vòng quyết_định là các đảng_viên đảng Bảo_thủ chứ không phải 358 nghị_sĩ của đảng này . Kết_quả cuộc thăm_dò dư_luận của hãng YouGov thực_hiện ngay trong tối 20/7 dự_đoán , ông Sunak có_thể sẽ chỉ giành được 35% phiếu ủng_hộ , so với 54% phiếu dành cho bà Liz_Truss . Website của đảng Bảo_thủ cũng tiến_hành một cuộc thăm_dò dư_luận riêng với các đảng_viên đảng này và cũng cho ra kết_quả tương_tự , rằng tỉ_lệ phiếu giữa bà Liz_Truss và ông Rishi_Sunak là 49% - 42% . CNN dẫn nhận_định của giới quan_sát cho_hay , dù ai là người chiến_thắng trong cuộc đua để trở_thành lãnh_đạo mới của đảng Bảo_thủ và đảm_nhiệm cương_vị Thủ_tướng Anh thay ông Boris_Johnson , nội_các mới của đảng Bảo_thủ sẽ vẫn đối_mặt với vô_vàn thách_thức lớn trong thời_gian tới do sự sụt_giảm tín_nhiệm trước công_chúng Anh .<doc-sep>Theo kênh_truyền_hình RT , bản kiến_nghị đã được hai đảng_viên cấp cao , là cựu nghị_sĩ David Campbell-Bannerman và nhà tỷ_phú Lord_Peter_Cruddas , trình lên chủ_tịch đảng Bảo_thủ . Bản kiến_nghị cho rằng các đảng_viên “ trung_thành và nỗ_lực ” đã bầu cho ông Johnson trong năm 2019 và những cử_tri trung_thành này không nên bị tước quyền lên_tiếng . Văn_bản cho_hay một_số thành_viên của đảng rất khó_chịu và tức_giận đối_với các nghị_sĩ của đảng . “ Tôi yêu_cầu tên ông Boris_Johnson phải được thêm vào danh_sách ứng_cử cho các nghị_sĩ bỏ_phiếu trong cuộc bầu_cử sắp tới ” , bản kiến_nghị với hơn 4.000 người ký_tên nêu rõ . Những người đứng đằng sau bản kiến_nghị cho rằng toàn_bộ quá_trình bầu nhà_lãnh_đạo mới của đảng đã mở_đường cho các nghị_sĩ có_thể viện lý_do lợi_ích để dàn_xếp chống lại Thủ_tướng Johnson . Campbell-Bannerman , người đứng đầu hai hiệp_hội đảng Bảo_thủ , chỉ ra hành_động loại_bỏ một nhà_lãnh_đạo đã từng được bầu_chọn sẽ là hành_động tự_sát đối_với những người trong đảng . Trong khi đó , nhà tỷ_phú Cruddas đe_doạ rút khoản đóng_góp 500.000 bảng Anh nếu_như tên ông Johnson không xuất_hiện trên lá phiếu . Tuy_nhiên , một phát_ngôn_viên của đảng Bảo_thủ giải_thích : “ Quy_định bầu_cử nêu rõ một_khi một nhà_lãnh_đạo đã từ_chức , họ sẽ không đủ điều_kiện để tham_gia lại trong một cuộc tranh_cử tiếp_theo ” . Một kiến_nghị khác nêu ý_kiến tương_tự đã bị Quốc_hội bác_bỏ vào ngày 21/7 . Đơn kiến_nghị được đệ_trình vào ngày 8/7 kêu_gọi tiến_hành một cuộc điều_tra khẩn_cấp về toàn_bộ quá_trình buộc Thủ_tướng phải từ_chức . Hiện cuộc đua kế_nhiệm Thủ_tướng Boris_Johnson đã rút xuống còn 2 ứng_viên , là cựu Bộ_trưởng Tài_chính Rishi_Sunak và Bộ_trưởng Ngoại_giao Liz_Truss . Người chiến_thắng sẽ được công_bố vào ngày 5/9 sau khi khoảng 150.000 thành_viên đảng Bảo_thủ bỏ_phiếu qua thư . Nhà_lãnh_đạo mới của đảng nghiễm_nhiên trở_thành thủ_tướng Anh . Cho đến khi chính_quyền mới thành_lập , Thủ_tướng sắp mãn_nhiệm Johnson vẫn sẽ tiếp_tục giữ chức và điều_hành đất_nước . Cuộc bầu nhà_lãnh_đạo mới của đảng được thông_báo sau khi ông Johnson quyết_định từ_chức sau một loạt bê_bối châm ngòi làn_sóng từ_chức của các thành_viên nội_các .<doc-sep>Hôm 21-7 , trang_web chính_thức của đảng Bảo_thủ Anh đã đăng_tải một bản kiến_nghị do hơn 4.000 cử_tri ủng_hộ đảng gửi lên yêu_cầu cho Thủ_tướng Boris_Johnson có cơ_hội thứ hai và tham_gia cuộc tranh_cử_chức thủ_tướng sắp tới . Đơn kiến_nghị , do cựu thành_viên Nghị_viện Châu_Âu David Campbell-Bannerman và tỉ_phú Peter_Cruddas ( thành_viên Thượng_Nghị_viện Vương_quốc Liên_hiệp Anh và Bắc_Ireland ) đệ_trình , đã được gửi lên Chủ_tịch đảng Bảo_thủ . Bản kiến_nghị nói rằng vì ông Johnson thắng_cử vào năm 2019 nhờ phiếu bầu của các cử_tri " trung_thành và chăm_chỉ " của đảng Bảo_thủ , nên cơ_bản họ có quyền được nói lên ý_kiến của mình . Các cử_tri cho biết họ cũng “ rất khó_chịu ” và “ tức_giận trước quyết_định của Quốc_hội Anh ” . “ Chúng_tôi yêu_cầu ông Boris_Johnson được thêm vào danh_sách ứng_cử_viên để các cử_tri bỏ_phiếu trong cuộc bầu_cử sắp tới ” - đơn kiến_nghị viết . Theo các cử_tri ký bản kiến_nghị , toàn_bộ quá_trình bầu một nhà_lãnh_đạo mới sẽ tạo cơ_hội để “ các nghị_sĩ lợi_dụng , những người có_thể viện lý_do lợi_ích của đất_nước và sự bất_bình ” để dàn_xếp chống lại ông Johnson . Theo ông Campbell-Bannerman , việc loại_bỏ ông Johnson là một “ hành_động tự_sát ” đối_với đảng Bảo_thủ , trong khi tỉ_phú Cruddas đe_doạ sẽ thu lại khoản đóng_góp 500.000 bảng Anh ( hơn 14 tỉ đồng ) trừ_phi tên của ông Johnson xuất_hiện trên lá phiếu . Tuy_nhiên , một phát_ngôn_viên của đảng Bảo_thủ cho biết “ luật_pháp đã nêu rõ rằng một_khi một nhà_lãnh_đạo đã từ_chức , họ sẽ không đủ điều_kiện để tham_gia cuộc tranh_cử tiếp_theo ” . Trong khi ông Campbell-Bannerman và ông Cruddas đe_doạ sẽ nộp đơn kêu_gọi điều_tra khẩn_cấp cuộc bầu_cử sắp tới nếu đơn kiến_nghị của họ không được chấp_thuận , một đơn kiến_nghị khác về vấn_đề tương_tự đã bị Quốc_hội Anh bác_bỏ vào ngày 21-7 . Đơn kháng_cáo , được đệ_trình vào ngày 8-7 , kêu_gọi " một cuộc điều_tra khẩn_cấp về toàn_bộ quá_trình buộc Thủ_tướng Johnson phải từ_chức . " “ Đưa ông Boris_Johnson trở_lại làm Thủ_tướng Anh . Ông ấy đã bị buộc rời khỏi vị_trí lãnh_đạo một_cách không công_bằng và ông ấy cần phải hoàn_thành nhiệm_vụ của mình vì lợi_ích của nước Anh ” - bản kiến_nghị viết . Trước đó , Cựu_Bộ_trưởng Tài_chính Rishi_Sunak và Ngoại_trưởng Liz_Truss đã trở_thành hai ứng_cử_viên cuối_cùng được các nghị_sĩ đảng Bảo_thủ lựa_chọn là người thay_thế ông Johnson sau khi vòng bỏ_phiếu thứ 5 bầu lãnh_đạo đảng cầm_quyền kết_thúc vào chiều 20-7 ( theo giờ_địa_phương ) . Ông Sunak và bà Truss sẽ thực_hiện các chiến_dịch vận_động tranh_cử trong toàn đảng trên cả nước trước khi 160.000 thành_viên đảng Bảo_thủ đưa ra quyết_định . Các thành_viên đảng Bảo_thủ và cử_tri cũng sẽ có cơ_hội lắng_nghe tranh_luận của hai ứng_viên trong chương_trình kéo_dài 60 phút , được phát_sóng trực_tiếp trên đài_truyền_hình BBC vào tối 25-7 ( theo giờ_địa_phương ) . Sau vòng bỏ_phiếu cuối_cùng , giờ_đây các đảng_viên Bảo_thủ sẽ đưa ra quyết_định lựa_chọn người đứng đầu_đảng cầm_quyền bằng lá phiếu bầu qua đường bưu_điện trong mùa hè này . Cuộc bỏ_phiếu toàn đảng sẽ kết_thúc vào ngày 2-9 và tên người chiến_thắng sẽ được công_bố vào ngày 5-9 . Lãnh_đạo đảng mới sẽ được bổ_nhiệm làm thủ_tướng ngay sau đó . Cho đến khi người đứng đầu chính_phủ mới được bổ_nhiệm , ông Johnson sẽ tiếp_tục giữ vai_trò thủ_tướng Anh .'
        # docs="Theo lời khai của Huy tại phiên_toà , để có tiền sử_dụng cá_nhân , Huy “ nổ ” là sĩ_quan cục Phòng_chống ma_tuý của bộ Công_an đóng tại TP. Đà_Nẵng , có nguồn mua ô_tô thanh_lý giá rẻ , và khả_năng chạy việc vào ngành công_an . Chỉ với lời “ nổ ” này , từ tháng 10/2016 đến 9/2017 , nhiều người đã bị lừa_đảo với tổng_số tiền 3,2 tỷ đồng . Trong đó , người bị Huy lừa nhiều nhất là vợ_chồng ông Bảo_Th . , ngụ quận Hải_Châu . Huy giới_thiệu với cặp vợ_chồng này mình có suất mua ô_tô thanh_lý giá rẻ và rủ họ mua cùng . Tin lời , vợ_chồng ông Th . đưa cho Huy hơn 1 tỷ đồng . Cùng thủ_đoạn , Huy lừa thêm ông Nguyễn_Tấn_T. 970 triệu đồng , Lê_Quốc_Th . 400 triệu đồng , Trần_Nhật_S. 300 triệu đồng … Sau chiêu_thức mua xe thanh_lý , Huy chuyển sang giả_vờ có khả_năng xin việc vào ngành công_an . Với chiêu_thức này , Huy lừa vợ_chồng ông Đinh_Ngọc_H. 250 triệu đồng . Ngoài_ra , Huy hứa_hẹn , tháng 3/2017 sẽ đưa kết_quả cho con ông H. đi làm_việc . Tuy_nhiên , sau nhiều lần hẹn mà không có quyết_định tuyển_dụng , ông H. đã gửi đơn tố_cáo đến cơ_quan Công_an . Từ đó , những hành_vi sai_trái của Huy lần_lượt được truy ra ."
        # docs = input("Enter documents separated by <doc-sep>:\n")
        docs = docs.split('<doc-sep>')
        # print(docs)
        #You can change the input to segmented
        # docs = [' '.join(rdrsegmenter.word_segment(doc)) for doc in docs]
        # print(docs)
        dataset = [{"document": docs, "summary": "_"}]
        print(dataset)
        print("Loading PRIMER model ...")
        # model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
        test_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "test", args.num_workers, False
        )
        args.compute_rouge = True
        # initialize trainer
        trainer = pl.Trainer(
            gpus=args.gpus,
            track_grad_norm=-1,
            max_steps=args.total_steps * args.acc_batch,
            replace_sampler_ddp=False,
            log_every_n_steps=5,
            checkpoint_callback=True,
            progress_bar_refresh_rate=args.progress_bar_refresh_rate,
            precision=32 if args.fp32 else 16,
            accelerator=args.accelerator,
            limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1
        )
        trainer.test(model, test_dataloader)

def initialize_model(args,model,docs):
    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
    # tokenizer.add_special_tokens(
    #             {"additional_special_tokens": ["<doc-sep>"]}
    # )
    
    ##Input
    # docs = input("Enter documents separated by <doc-sep>:\n")
    # docs='Với 349 phiếu ủng_hộ so với 238 phiếu chống , chính_phủ của Thủ_tướng Boris_Johnson ngày 18/7 ( giờ_địa_phương ) đã vượt qua cuộc bỏ_phiếu tín_nhiệm tự đề_xuất sau khi các thành_viên đảng Bảo_thủ cầm_quyền buộc phải lựa_chọn ủng_hộ chính_phủ nhằm tránh bầu_cử sớm . Trước đó , Công đảng đối_lập đã kêu_gọi thay_thế ông Boris_Johnson ngay_lập_tức bằng một thủ_tướng tạm_quyền cho đến khi người kế_nhiệm chính_thức được đảng Bảo_thủ bầu ra vào đầu tháng 9 . Công đảng đã tìm cách tổ_chức bỏ_phiếu tín_nhiệm đối_với cả chính_phủ và cá_nhân ông Boris_Johnson nhằm buộc ông phải rút_lui sớm hơn . Tuy_nhiên , Chính_phủ Anh phản_đối với lý_do như_vậy là không cần_thiết sau khi Thủ_tướng Boris_Johnson đã cam_kết sẽ rời bỏ chức_vụ khi có người thay_thế . Thay vào đó , đảng Bảo_thủ đề_xuất chỉ bỏ_phiếu tín_nhiệm đối_với chính_phủ . Ngày 7/7 vừa_qua , ông Boris_Johnson đã thông_báo từ_chức lãnh_đạo đảng Bảo_thủ , đồng_nghĩa với việc từ_chức Thủ_tướng Anh , trong bối_cảnh nhà_lãnh_đạo này vướng phải hàng_loạt bê_bối và đối_mặt với làn_sóng từ_chức của hàng chục thành_viên cấp cao trong chính_phủ . Tuy_nhiên , ông Boris_Johnson sẽ vẫn tiếp_tục vai_trò lãnh_đạo Chính_phủ Anh cho đến khi đảng Bảo_thủ bầu được lãnh_đạo mới . Các thành_viên trong đảng Bảo_thủ kỳ_vọng quá_trình bầu_cử nội_bộ 2 giai_đoạn sẽ hoàn_thành trước khi diễn ra hội_nghị thường_niên của đảng vào tháng 10 tới . Thông_báo từ_nhiệm của ông Boris_Johnson được đánh_giá là một quyết_định quan_trọng và trong trường_hợp này càng đáng chú_ý hơn . Trên thực_tế , ông Boris_Johnson không hề “ thất_cử ” . Tháng 12/2019 , ông đã lãnh_đạo Đảng Bảo_thủ giành chiến_thắng quyết_đoán trong tổng_tuyển_cử . Chính_phủ Bảo_thủ có được thế đa_số đáng_kể tại Hạ_viện và vài tuần trước ông đã vượt qua dễ_dàng cuộc bỏ_phiếu tín_nhiệm nội_bộ đảng . Tuy_nhiên , ông đang đi ngược_lại chiều gió bất_chấp việc có những món quà lớn nhất mà Hiến_pháp Anh có_thể “ ban_tặng ” cho một thủ_tướng - đa_số đủ để giúp thông_qua các dự_luật tại Hạ_viện và vị_thế để thúc_đẩy điều tương_tự ở Thượng_viện . Việc ông quyết_định từ_chức cho thấy nhiều điều trong cả quyền_lực và điểm yếu của vị_trí thủ_tướng . Nhìn bên ngoài , Văn_phòng Thủ_tướng Anh là nơi có sức_mạnh to_lớn . Về lý_thuyết , Thủ_tướng và đảng của họ chỉ tuân_thủ luật của các cuộc tổng_tuyển_cử . Tuy_nhiên , kể từ năm 1974 , các thủ_tướng có xu_hướng lên nắm quyền hoặc mất chức giữa các cuộc tổng_tuyển_cử , hoặc cả 2 với trường_hợp của bà Theresa_May và ông Boris_Johnson . Điều này là do ghế thủ_tướng được chọn_lựa phụ_thuộc vào sự tín_nhiệm của cả nội_các và đa_số nghị_sỹ Quốc_hội của chính_đảng nắm thế đa_số . Khi niềm tin không còn , các nhà_lãnh_đạo thường có xu_hướng từ_chức . Việc các thành_viên của đảng mất niềm tin có_thể thể_hiện qua nhiều cách ngoài lá phiếu tín_nhiệm , chẳng_hạn như lời kêu_gọi bầu_cử hoặc bằng cách từ_chức và từ_chối được bổ_nhiệm . Một thủ_tướng có_thể tìm cách phớt_lờ những tín_hiệu này , song tới cuối_cùng họ sẽ phải đối_mặt với các cuộc bỏ_phiếu bất_tín_nhiệm chính_thức của Quốc_hội hoặc bị Nữ_hoàng Elizabeth cách_chức . Có những đồn đoán rằng ông Johnson sẽ “ chiến_đấu ” sau hàng_loạt đơn_từ_chức bất_thường và những lá thư bất_tín_nhiệm trong vài ngày qua . Song thực_tế mọi chuyện đã khác . Trong cuộc bỏ_phiếu tín_nhiệm nội_bộ của đảng Bảo_thủ , ông nhận được 211 phiếu , hơn ngưỡng 180 cần_thiết để giành chiến_thắng . Tuy_nhiên , ông đã thua theo một nghĩa chính_trị rộng_lớn hơn , quan_trọng hơn , ông không thắng đủ để chấm_dứt những câu hỏi về năng_lực lãnh_đạo đã đeo_bám ông từ lâu . Ví_dụ , trong phần_lớn cuộc khủng_hoảng COVID-19 , cách tiếp_cận để giải_quyết vấn_đề của ông đa_phần hỗn_loạn và thiếu nhất_quán , phản_ánh một thực_tế là phong_cách của ông thiên về “ bức tranh lớn ” chứ không tập_trung vào chi_tiết , trong khi phong_cách sống lại không mấy phù_hợp với các đòi_hỏi của thời_đại dịch . Tuy_nhiên , những người ủng_hộ ông Boris_Johnson sẽ lập_luận rằng ông đã đạt được hai chiến_thắng quan_trọng . Thứ nhất là chiến_thắng trong cuộc tổng_tuyển_cử mang tính bước_ngoặt vào năm 2019 , khi Đảng Bảo_thủ ít_nhất cũng tạm_thời đã vẽ lại bản_đồ chính_trị của Anh bằng cách giành được một_số thành_trì lâu_đời của Công đảng , đặc_biệt là ở Midlands và Bắc_Anh . Thứ hai , ông có công lớn trong cuộc trưng_cầu_ý_dân về Brexit gây chia_rẽ vào năm 2016 và sau đó đưa Anh rời khỏi Liên_minh châu_Âu ( EU ) vào năm 2020 . Đây là lý_do tại_sao di_sản chính_trị của nhà_lãnh_đạo này có_thể sẽ là đề_tài gây tranh_cãi trong nhiều năm tới . Dù đã đạt được những thành_tựu lịch_sử , bao_gồm cả Brexit , quốc_gia này chắc_chắn sẽ bị chia_rẽ nhiều hơn là đoàn_kết bởi nhiệm_kỳ thủ_tướng gây tranh_cãi của ông Boris_Johnson . Ông Boris_Johnson có_thể đã từ_chức lãnh_đạo Đảng Bảo_thủ nhưng những rắc_rối của đảng cầm_quyền ở Anh còn lâu mới kết_thúc . Trong khi các nghị_sỹ Đảng Bảo_thủ phần_lớn đồng_tình rằng giờ là thời_điểm thích_hợp để Thủ_tướng Johnson ra đi , chính_đảng này đang rất chia_rẽ về việc người kế_nhiệm ông nên là ai . Nhiều khả_năng sẽ có những cuộc tranh_luận gay_gắt trong đảng những tuần tới , về cả việc liệu nước này có cần quay lại hướng tiếp_cận áp mức thuế thấp theo kiểu Thatcher , một_cách tiếp_cận nhà_nước quy_mô hẹp hơn sau khi lựa_chọn cách can_thiệp kiểu ông Boris_Johnson với các khoản chi_tiêu lớn , ảnh_hưởng bởi tình_hình khẩn_cấp về y_tế và kinh_tế trong giai_đoạn đại_dịch hay không . Sẽ có rất nhiều ứng_cử_viên cho vị_trí lãnh_đạo của đảng và nhân_vật sẽ ngồi vào ghế thủ_tướng . Các ứng_cử_viên đã được công_bố gồm Tổng_Chưởng lý Suella_Braverman và Chủ_tịch Uỷ_ban Đối_ngoại Tom_Tugendhat và dự_kiến những cái tên khác như Ngoại_trưởng Liz_Truss ; cựu Bộ_trưởng Giáo_dục Nadhim_Zahawi , người vừa được Johnson bổ_nhiệm làm Bộ_trưởng Tài_chính hôm 4/7 ; Bộ_trưởng Thương_mại Penny_Mordaunt , Bộ_trưởng Quốc_phòng Ben_Wallace và Bộ_trưởng Tư_pháp Dominic_Raab . Còn những ứng_cử_viên tiềm_năng khác như cựu Bộ_trưởng Y_tế Sajid_Javid và cựu Bộ_trưởng Tài_chính Rishi_Sunak , những người bất_ngờ từ_chức hôm 4/7 . Nhiệm_vụ quan_trọng đối_với việc tìm người thay_thế ông Boris_Johnson sẽ là chấm_dứt tình_trạng tê_liệt trong chính_phủ bằng một tầm nhìn rõ_ràng hậu Brexit trong quá_trình hồi_phục sau đại_dịch .<doc-sep>Theo thông_báo của đảng Bảo_thủ , cựu Bộ_trưởng Tài_chính Rishi_Sunak là người về nhất trong vòng bỏ_phiếu vừa_qua với 137 phiếu ủng_hộ . Trong khi đó , Ngoại_trưởng Liz_Truss đánh_bại Bộ_trưởng Thương_mại Penny_Mordaunt để giành tấm vé vào vòng cuối_cùng với tỉ_lệ phiếu 113 - 105 . Sau khi kết_quả được công_bố , cả hai ứng_viên đã chia_sẻ quan_điểm trên trang Twitter cá_nhân . " Xin gửi lời cảm_ơn tới các cộng_sự đã đặt niềm tin vào tôi . Tôi sẽ làm_việc ngày_đêm để truyền đi thông_điệp của chúng_ta tới thế_giới " , ông Sunak viết . Về phía Ngoại_trưởng Liz_Truss , bà nhấn_mạnh luôn sẵn_sàng và nỗ_lực để đạt được mục_tiêu từ ngày đầu của cuộc chạy_đua . Như_vậy , ông Rishi_Sunak và bà Liz_Truss sẽ đối_mặt trong vòng bỏ_phiếu quyết_định cuối_cùng của trên 170.000 đảng_viên đảng Bảo_thủ , dự_kiến tổ_chức từ cuối tháng 8 - 2/9 . Trước_mắt , hai ứng_viên này sẽ có phiên tranh_luận trực_tiếp trên truyền_hình vào ngày 25/7 và sẽ tiến_hành các chiến_dịch vận_động tranh_cử trong tháng 8 . Theo nhận_định của truyền_thông Anh , không_thể đoán trước kết_quả cuộc đua giữa ông Rishi_Sunak và bà Liz_Truss . Thực_tế cho thấy ông Sunak thường về nhất trong các vòng bỏ_phiếu vừa_qua , nhưng bà Liz_Truss được cho là có lợi_thế hơn ở vòng cuối bởi những người bỏ_phiếu trong vòng quyết_định là các đảng_viên đảng Bảo_thủ chứ không phải 358 nghị_sĩ của đảng này . Kết_quả cuộc thăm_dò dư_luận của hãng YouGov thực_hiện ngay trong tối 20/7 dự_đoán , ông Sunak có_thể sẽ chỉ giành được 35% phiếu ủng_hộ , so với 54% phiếu dành cho bà Liz_Truss . Website của đảng Bảo_thủ cũng tiến_hành một cuộc thăm_dò dư_luận riêng với các đảng_viên đảng này và cũng cho ra kết_quả tương_tự , rằng tỉ_lệ phiếu giữa bà Liz_Truss và ông Rishi_Sunak là 49% - 42% . CNN dẫn nhận_định của giới quan_sát cho_hay , dù ai là người chiến_thắng trong cuộc đua để trở_thành lãnh_đạo mới của đảng Bảo_thủ và đảm_nhiệm cương_vị Thủ_tướng Anh thay ông Boris_Johnson , nội_các mới của đảng Bảo_thủ sẽ vẫn đối_mặt với vô_vàn thách_thức lớn trong thời_gian tới do sự sụt_giảm tín_nhiệm trước công_chúng Anh .<doc-sep>Theo kênh_truyền_hình RT , bản kiến_nghị đã được hai đảng_viên cấp cao , là cựu nghị_sĩ David Campbell-Bannerman và nhà tỷ_phú Lord_Peter_Cruddas , trình lên chủ_tịch đảng Bảo_thủ . Bản kiến_nghị cho rằng các đảng_viên “ trung_thành và nỗ_lực ” đã bầu cho ông Johnson trong năm 2019 và những cử_tri trung_thành này không nên bị tước quyền lên_tiếng . Văn_bản cho_hay một_số thành_viên của đảng rất khó_chịu và tức_giận đối_với các nghị_sĩ của đảng . “ Tôi yêu_cầu tên ông Boris_Johnson phải được thêm vào danh_sách ứng_cử cho các nghị_sĩ bỏ_phiếu trong cuộc bầu_cử sắp tới ” , bản kiến_nghị với hơn 4.000 người ký_tên nêu rõ . Những người đứng đằng sau bản kiến_nghị cho rằng toàn_bộ quá_trình bầu nhà_lãnh_đạo mới của đảng đã mở_đường cho các nghị_sĩ có_thể viện lý_do lợi_ích để dàn_xếp chống lại Thủ_tướng Johnson . Campbell-Bannerman , người đứng đầu hai hiệp_hội đảng Bảo_thủ , chỉ ra hành_động loại_bỏ một nhà_lãnh_đạo đã từng được bầu_chọn sẽ là hành_động tự_sát đối_với những người trong đảng . Trong khi đó , nhà tỷ_phú Cruddas đe_doạ rút khoản đóng_góp 500.000 bảng Anh nếu_như tên ông Johnson không xuất_hiện trên lá phiếu . Tuy_nhiên , một phát_ngôn_viên của đảng Bảo_thủ giải_thích : “ Quy_định bầu_cử nêu rõ một_khi một nhà_lãnh_đạo đã từ_chức , họ sẽ không đủ điều_kiện để tham_gia lại trong một cuộc tranh_cử tiếp_theo ” . Một kiến_nghị khác nêu ý_kiến tương_tự đã bị Quốc_hội bác_bỏ vào ngày 21/7 . Đơn kiến_nghị được đệ_trình vào ngày 8/7 kêu_gọi tiến_hành một cuộc điều_tra khẩn_cấp về toàn_bộ quá_trình buộc Thủ_tướng phải từ_chức . Hiện cuộc đua kế_nhiệm Thủ_tướng Boris_Johnson đã rút xuống còn 2 ứng_viên , là cựu Bộ_trưởng Tài_chính Rishi_Sunak và Bộ_trưởng Ngoại_giao Liz_Truss . Người chiến_thắng sẽ được công_bố vào ngày 5/9 sau khi khoảng 150.000 thành_viên đảng Bảo_thủ bỏ_phiếu qua thư . Nhà_lãnh_đạo mới của đảng nghiễm_nhiên trở_thành thủ_tướng Anh . Cho đến khi chính_quyền mới thành_lập , Thủ_tướng sắp mãn_nhiệm Johnson vẫn sẽ tiếp_tục giữ chức và điều_hành đất_nước . Cuộc bầu nhà_lãnh_đạo mới của đảng được thông_báo sau khi ông Johnson quyết_định từ_chức sau một loạt bê_bối châm ngòi làn_sóng từ_chức của các thành_viên nội_các .<doc-sep>Hôm 21-7 , trang_web chính_thức của đảng Bảo_thủ Anh đã đăng_tải một bản kiến_nghị do hơn 4.000 cử_tri ủng_hộ đảng gửi lên yêu_cầu cho Thủ_tướng Boris_Johnson có cơ_hội thứ hai và tham_gia cuộc tranh_cử_chức thủ_tướng sắp tới . Đơn kiến_nghị , do cựu thành_viên Nghị_viện Châu_Âu David Campbell-Bannerman và tỉ_phú Peter_Cruddas ( thành_viên Thượng_Nghị_viện Vương_quốc Liên_hiệp Anh và Bắc_Ireland ) đệ_trình , đã được gửi lên Chủ_tịch đảng Bảo_thủ . Bản kiến_nghị nói rằng vì ông Johnson thắng_cử vào năm 2019 nhờ phiếu bầu của các cử_tri " trung_thành và chăm_chỉ " của đảng Bảo_thủ , nên cơ_bản họ có quyền được nói lên ý_kiến của mình . Các cử_tri cho biết họ cũng “ rất khó_chịu ” và “ tức_giận trước quyết_định của Quốc_hội Anh ” . “ Chúng_tôi yêu_cầu ông Boris_Johnson được thêm vào danh_sách ứng_cử_viên để các cử_tri bỏ_phiếu trong cuộc bầu_cử sắp tới ” - đơn kiến_nghị viết . Theo các cử_tri ký bản kiến_nghị , toàn_bộ quá_trình bầu một nhà_lãnh_đạo mới sẽ tạo cơ_hội để “ các nghị_sĩ lợi_dụng , những người có_thể viện lý_do lợi_ích của đất_nước và sự bất_bình ” để dàn_xếp chống lại ông Johnson . Theo ông Campbell-Bannerman , việc loại_bỏ ông Johnson là một “ hành_động tự_sát ” đối_với đảng Bảo_thủ , trong khi tỉ_phú Cruddas đe_doạ sẽ thu lại khoản đóng_góp 500.000 bảng Anh ( hơn 14 tỉ đồng ) trừ_phi tên của ông Johnson xuất_hiện trên lá phiếu . Tuy_nhiên , một phát_ngôn_viên của đảng Bảo_thủ cho biết “ luật_pháp đã nêu rõ rằng một_khi một nhà_lãnh_đạo đã từ_chức , họ sẽ không đủ điều_kiện để tham_gia cuộc tranh_cử tiếp_theo ” . Trong khi ông Campbell-Bannerman và ông Cruddas đe_doạ sẽ nộp đơn kêu_gọi điều_tra khẩn_cấp cuộc bầu_cử sắp tới nếu đơn kiến_nghị của họ không được chấp_thuận , một đơn kiến_nghị khác về vấn_đề tương_tự đã bị Quốc_hội Anh bác_bỏ vào ngày 21-7 . Đơn kháng_cáo , được đệ_trình vào ngày 8-7 , kêu_gọi " một cuộc điều_tra khẩn_cấp về toàn_bộ quá_trình buộc Thủ_tướng Johnson phải từ_chức . " “ Đưa ông Boris_Johnson trở_lại làm Thủ_tướng Anh . Ông ấy đã bị buộc rời khỏi vị_trí lãnh_đạo một_cách không công_bằng và ông ấy cần phải hoàn_thành nhiệm_vụ của mình vì lợi_ích của nước Anh ” - bản kiến_nghị viết . Trước đó , Cựu_Bộ_trưởng Tài_chính Rishi_Sunak và Ngoại_trưởng Liz_Truss đã trở_thành hai ứng_cử_viên cuối_cùng được các nghị_sĩ đảng Bảo_thủ lựa_chọn là người thay_thế ông Johnson sau khi vòng bỏ_phiếu thứ 5 bầu lãnh_đạo đảng cầm_quyền kết_thúc vào chiều 20-7 ( theo giờ_địa_phương ) . Ông Sunak và bà Truss sẽ thực_hiện các chiến_dịch vận_động tranh_cử trong toàn đảng trên cả nước trước khi 160.000 thành_viên đảng Bảo_thủ đưa ra quyết_định . Các thành_viên đảng Bảo_thủ và cử_tri cũng sẽ có cơ_hội lắng_nghe tranh_luận của hai ứng_viên trong chương_trình kéo_dài 60 phút , được phát_sóng trực_tiếp trên đài_truyền_hình BBC vào tối 25-7 ( theo giờ_địa_phương ) . Sau vòng bỏ_phiếu cuối_cùng , giờ_đây các đảng_viên Bảo_thủ sẽ đưa ra quyết_định lựa_chọn người đứng đầu_đảng cầm_quyền bằng lá phiếu bầu qua đường bưu_điện trong mùa hè này . Cuộc bỏ_phiếu toàn đảng sẽ kết_thúc vào ngày 2-9 và tên người chiến_thắng sẽ được công_bố vào ngày 5-9 . Lãnh_đạo đảng mới sẽ được bổ_nhiệm làm thủ_tướng ngay sau đó . Cho đến khi người đứng đầu chính_phủ mới được bổ_nhiệm , ông Johnson sẽ tiếp_tục giữ vai_trò thủ_tướng Anh .'
    # docs=''
    try:
        docs = docs.split('<doc-sep>')
    except Exception:
        pass
    # print(docs)
    #You can change the input to segmented
    # docs = [' '.join(rdrsegmenter.word_segment(doc)) for doc in docs]
    dataset = [{"document": docs, "summary": "_"}]
    # print(dataset)
    # print("Loading PRIMER model ...")
    test_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "test", args.num_workers, False
    )

    for (idx, batch) in enumerate(test_dataloader):
        # print(batch)
        input_ids,output_ids,tgt = batch
        print(input_ids.size(),output_ids.size())
        break
    
    #generate
    input_ids, attention_mask = model._prepare_input(input_ids)
    print("Predicting...")
    generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=args.max_length_tgt,
            min_length=args.min_length_tgt,
            num_beams=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=3 ,
    )
    generated_str = model.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
    )

    #write file
    output_dir = os.path.join(
            args.model_path,
            "predicted_folder"    
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for pred in generated_str:

        with open(os.path.join(output_dir, "prediction.txt"), "w", encoding="utf-8") as of:
            of.write(pred)
        with open(os.path.join(output_dir, "prediction.jsonl_5" ), "w", encoding="utf-8") as fo:
            json.dump(pred, fo, ensure_ascii=False, indent=4)
        break
    print(generated_str,len(generated_str))
    return generated_str