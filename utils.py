import argparse
import pickle
import itertools
import linecache
import torch
import logging
import os
import json
import numpy as np
import pytorch_lightning as pl
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Union
from pathlib import Path
from rouge_score import rouge_scorer, scoring
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from transformers import (
    BartTokenizer,
)

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

try:
    from fairseq.data.data_utils import batch_by_size
    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


class Seq2SeqLoggingCallback(pl.Callback):

    logger = logging.getLogger(__name__)

    """
    def on_train_batch_end(self, trainer, pl_module):
        return
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)
    """

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print('call seq2seq callback. ')
        self.gen_path = Path(pl_module.output_dir) / 'generate_result'
        os.makedirs(self.gen_path, exist_ok=True)
        return
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_validation_start(self, trainer: pl.Trainer, pl_module):
        print('start validation. ')
        self.gen_path = Path(pl_module.output_dir) / 'generate_result'
        os.makedirs(self.gen_path, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        self._generate_text(trainer, pl_module)
        # Uncommenting this will save val generations
        # return self._write_logs(trainer, pl_module, "valid")

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        return
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def _generate_text(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if len(pl_module.validation_result) > 0:
            gen_file_path = Path(self.gen_path) / f'gen_result_epoch={trainer.current_epoch}.json'
            with open(gen_file_path, 'w') as f:
                json.dump(pl_module.validation_result, f, indent=4)
            pl_module.validation_result = []

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        self.logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        # period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )

def get_scheduler_info():
    # update this and the import above to support new schedulers from transformers.optimization
    arg_to_scheduler = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
        "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
        "polynomial": get_polynomial_decay_schedule_with_warmup,
        # '': get_constant_schedule,             # not supported for now
        # '': get_constant_schedule_with_warmup, # not supported for now
    }
    arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
    arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

    scheduler_dict = dict(
        arg_to_scheduler=arg_to_scheduler,
        arg_to_scheduler_choices=arg_to_scheduler_choices,
        arg_to_scheduler_metavar=arg_to_scheduler_metavar,
    )

    return scheduler_dict

def label_to_tensor(label: str):
    return torch.Tensor([int(label)])

def encode_line(tokenizer, line, max_length, dataset_name=None, pad_to_max_length=True, return_tensors="pt"):
    check_variable_status(dataset_name, name='dataset_name')
    is_nli = check_nli_dataset(dataset_name)
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    separater = '@separater@'
    texts = (sent.strip() for sent in line.split(separater)) if is_nli else (line, )
    return tokenizer(
        [*texts],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )

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
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return (input_ids[:, keep_column_mask], )
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], )

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def calculate_rouge(output_lns: List[str], reference_lns: List[str], rouge_keys: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def use_task_specific_params(model, task=None):
    """Update config with summarization specific params."""
    if task is not None:
        task_specific_params = model.config.task_specific_params

        if task_specific_params is not None:
            pars = task_specific_params.get(task, {})
            check_task_specific_params_type(pars)
            print(f"using task specific params for {task}: {pars}")
            model.config.update(pars)

def check_task_specific_params_type(pars):
    int_params = ['num_labels']
    float_params = []
    for param in int_params:
        if param in pars.keys():
            pars[param] = int(pars[param])
    for param in float_params:
        if param in pars.keys():
            pars[param] = float(pars[param])

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

''' check functions '''

def check_argument_setting(args, arg_name):
    if arg_name == 'task':
        assert args.task in ['agnews', 'mrpc', 'news-summary', 'wiki80']
    elif arg_name == 'model_mode':
        assert args.model_mode in ['base', 'sequence-classification', 'question-answering', \
            'pretraining', 'token-classification', 'language-modeling', \
            'summarization', 'translation']

def check_variable_status(variable, name="", status='None'):
    if status == 'None':
        if variable is None:
            raise ValueError('{} parameter should not be none. '.format(name))

def check_parameter_value(hparams, param_name_list, check_all=False):
    if isinstance(hparams, argparse.Namespace):
        hparams = vars(hparams)
    # not_none_param = ['model_mode', 'model_name_or_path', 'config_name', 'tokenizer_name']
    check_exist = False
    for param_name in param_name_list:
        if hparams[param_name] is None:
            if check_all:
                raise ValueError('{} parameter should not be none. '.format(param_name))
        else:
            check_exist = True
    if not check_exist:
        raise ValueError('paramters in list should have at least one not none value. ')

def check_nli_dataset(dataset_name):
    nli = []
    if dataset_name in nli:
        return True
    return False