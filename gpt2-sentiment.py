# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import copy
import argparse
import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    HfArgumentParser, 
    pipeline, 
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from dataset import GeneralDataset
from utils import (
    pickle_load, 
    pickle_save, 
)

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    do_train: bool = field(
        default=False, 
        metadata={"help": "do training if set to true. "}
    )
    do_eval: bool = field(
        default=False, 
        metadata={"help": "do evaluation if set to true. "}
    )
    train_type: Optional[str] = field(
        default=None, 
        metadata={"help": "the ppo model train path. "}
    )
    use_cls_type: Optional[str] = field(
        default=None, 
        metadata={"help": "the classifer model type. "}
    )
    dataset_type: Optional[str] = field(
        default=None, 
        metadata={"help": "the dataset type. "}
    )
    load_checkpoint_path: Optional[str] = field(
        default=None, 
        metadata={"help": "the load checkpoint path. "}
    )


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.
    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: 2] #input_size()]
        sample["input_ids"] = tokenizer.encode(
            sample["review"], 
            max_length=128, 
            truncation=True, 
            padding='max_length'
        )
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds

def collator(data):
    batch = dict((key, [d[key] for d in data]) for key in data[0])
    #for key in ['input_ids', 'label']:
    #    batch[key] = torch.stack(batch[key])
    return batch


class PPO(pl.LightningModule):
    def __init__(
        self, 
        args_list, 
        model_path, 
        dataset_path, 
        train_type='forward',
        use_cls_type='victim', 
        dataset_type='victim', 
    ):
        super().__init__()
        self.save_hparams(*args_list)
        self.hparams.global_epoch = 0
        hparams = self.hparams
        if hparams.output_dir is None:
            model_name = hparams.forward_model_name.replace('/', '-') if train_type == 'forward' \
                                                else hparams.backward_model_name.replace('/', '-')
            cls_name = hparams.victim_model_name.replace('/', '-') if use_cls_type == 'victim' \
                                                else hparams.attribute_model_name.replace('/', '-')
            dataset_name = hparams.victim_dataset_name.replace('/', '-') if dataset_type == 'victim' \
                                                else hparams.attribute_dataset_name.replace('/', '-')
            prefix = 'm={}_mt={}_cls={}_clst={}_ds={}'.format(model_name, train_type, cls_name, use_cls_type, dataset_name)
            output_dir_name = prefix 
            """ use if I want to tune these parameters
                        + '_tb={}_'.format(self.hparams.train_batch_size) + \
                        'e={}_'.format(self.hparams.num_train_epochs) + 'd={}_'.format(self.hparams.dropout) + \
                        'l={}_'.format(self.hparams.label_smoothing) + 'lr={}_'.format(self.hparams.learning_rate) \
                        + 'w={}_'.format(self.hparams.weight_decay) + 's={}'.format(self.hparams.seed)
            """
            self.hparams.output_dir = os.path.join(model_path, output_dir_name)
            Path(self.hparams.output_dir).mkdir(parents=True, exist_ok=True)

            if self.hparams.log_with == 'wandb':
                wandb.init(project="gpt2-sentiment-test2")#, entity='test1')
        # else:
            # self.output_dir = os.path.join(model_path, self.hparams.output_dir)

        if len(os.listdir(self.hparams.output_dir)) > 3:
            print('Output directory ({}) already exists and is not empty, may overwrite to it...'.format(self.hparams.output_dir))

        self.init_dataset(dataset_path, dataset_type)
        self.init_model(train_type, use_cls_type)

    def save_hparams(self, *args) -> argparse.Namespace:
        total_args = dict()
        for a in args: total_args.update(vars(a))
        self.save_hyperparameters(argparse.Namespace(**total_args))

    def init_dataset(self, dataset_path: str, dataset_type: str):
        dataset_name = self.hparams.victim_dataset_name if dataset_type == 'victim' \
                                else self.hparams.attribute_dataset_name
        dataset_path = dataset_name if self.hparams.data_load_from_hf \
                                                else os.path.join(dataset_path, dataset_name)
        self.dataset_kwargs: dict = dict(
            data_dir=dataset_path,
            max_source_length=self.hparams.max_source_length,
        )
        self.dataset_type = dataset_type
        self.dataset_class = GeneralDataset

    def init_model(self, train_type: str, use_cls_type: str):
        decoder_only = True
        self.train_type = train_type
        self.use_cls_type = use_cls_type
        self.ppo_config = PPOConfig(**self.ppo_config_kwargs)
        set_seed(self.ppo_config.seed)
        cls_model_name = self.hparams.victim_model_name if use_cls_type == 'victim' else self.hparams.attribute_model_name
        model_name = self.hparams.forward_model_name if train_type == 'forward' else self.hparams.backward_model_name
        ref_model_name = self.hparams.forward_ref_model_name if train_type == 'forward' else self.hparams.backward_ref_model_name

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_name)
        
        tokenizer_kwargs = dict(padding_side='left') if decoder_only else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hparams.pad_token_id = self.tokenizer.eos_token_id
        
        if not self.hparams.use_cls_pipeline:
            self.cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name)
            self.cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
        else:
            self.cls_model_name = cls_model_name
        


    @property 
    def model_generation_kwargs(self):
        args_dict = dict(self.hparams)
        kwargs_name = ['max_length', 'min_length', 'top_k', 'top_p', 'do_sample', 'pad_token_id', ]
        hparams_name = ['max_gen_target_length', 'min_gen_target_length', 'top_k', 'top_p', 'do_sample', 'pad_token_id', ]
        kwargs = {key: args_dict[value] for key, value in zip(kwargs_name, hparams_name)}
        # set max_length due to the warnings
        kwargs['max_length'] = 80
        return kwargs

    @property 
    def ppo_config_kwargs(self):
        args_dict = dict(self.hparams)
        kwargs_name = ['model_name', 'learning_rate', 'log_with', 'mini_batch_size', 
            'batch_size', 'gradient_accumulation_steps', 'seed', ]
        hparams_name = ['forward_model_name', 'learning_rate', 'log_with', 'mini_batch_size', 
            'train_batch_size', 'gradient_accumulation_steps', 'seed', ]
        kwargs = {key: args_dict[value] for key, value in zip(kwargs_name, hparams_name)}
        return kwargs

    def get_dataset(self, tokenizer=None, type_path='train'):
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        if self.hparams.data_load_from_hf:
            dataset = load_dataset(self.dataset_kwargs['data_dir'], split=type_path)
        elif tokenizer is None:
            dataset = self.dataset_class(
                self.tokenizer, 
                type_path=type_path,
                n_obs=n_obs[type_path],
                max_target_length=self.hparams.max_target_length,
                # task_mode='summarization', 
                task_name='task', # just make sure works for some checking func
                pad_to_max_len=self.hparams.pad_to_max_len, 
                **self.dataset_kwargs,
            )
        else:
            dataset = self.dataset_class(
                tokenizer, 
                type_path=type_path,
                n_obs=n_obs[type_path],
                max_target_length=self.hparams.max_target_length,
                # task_mode='summarization', 
                task_name='task', # just make sure works for some checking func
                pad_to_max_len=self.hparams.pad_to_max_len, 
                **self.dataset_kwargs,
            )
        return dataset
        # dataset = build_dataset(config)

    def train(self):
        dataset = self.get_dataset()
        ppo_trainer = PPOTrainer(
            self.ppo_config, 
            self.model, 
            self.ref_model, 
            self.tokenizer, 
            dataset=dataset, 
            data_collator=dataset.collate_fn, 
        )

        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
        # return_all_scores is deprecated. 
        sentiment_pipe = pipeline("sentiment-analysis", model=self.cls_model_name, device=device)
        sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
        print(f'the classifier model that is used to give reward is {self.cls_model_name}, type {self.use_cls_type}. ')

        def pad_to_same_len(query_tensor):
            query_list = query_tensor.tolist()
            if len(query_list) > self.hparams.max_gen_target_length: 
                return torch.tensor(query_list[:self.hparams.max_gen_target_length])
            else:
                return torch.tensor(query_list + [self.tokenizer.pad_token_id \
                            for i in range(self.hparams.max_gen_target_length - len(query_list) - 1)])

        for epoch in range(self.hparams.num_train_epochs):
            for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
                print(f'step: {step}')
                query_tensors = [query for query in batch["input_ids"]]
                # Get response from gpt2
                # kwargs not used: return_prompt=False, length_sampler=output_length_sampler,
                query_lists = [ppo_trainer.generate(query, **self.model_generation_kwargs).squeeze() \
                                                                                for query in query_tensors]
                # calculate reward
                # querys = torch.stack(query_lists)
                texts = [self.tokenizer.decode(query, skip_special_tokens=True) for query in query_lists]
                pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
                if self.use_cls_type == 'victim':
                    rewards = [torch.tensor(output[data.item()]["score"]) for output, data in zip(pipe_outputs, batch['labels'])]
                else: # attribute
                    rewards = [torch.tensor(output[self.hparams.attribute_id]["score"]) for output, data in zip(pipe_outputs, batch['labels'])]

                # Run PPO step
                stats = ppo_trainer.step(query_tensors, query_lists, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)

            if epoch % self.hparams.model_save_period == 0:
                self.save_checkpoint(epoch)

            

    def generate(
        self, 
        result_path='result', 
        type_path='val', 
        result_file='result.json', 
        attack_file='attack_result.json', 
        write_file=True, 
    ):
        # todo: write to file and log val metrics to wandb
        # todo: val metric: forward: victim model accuracy, perplexity, bleu with original sent, 
        # todo: val metric: backward: attribute model accuracy, victim model accuracy, perplexity, bleu with original sent, 
        #self.victim_model 
        #self.attr_model 
        self.hparams.n_val = -1

        victim_model = AutoModelForSequenceClassification.from_pretrained(self.hparams.victim_model_name)
        victim_tokenizer = AutoTokenizer.from_pretrained(self.hparams.victim_model_name)
        if self.train_type == 'backward':
            attr_model = AutoModelForSequenceClassification.from_pretrained(self.hparams.attribute_model_name)
            attr_tokenizer = AutoTokenizer.from_pretrained(self.hparams.attribute_model_name)

        dataset = self.get_dataset(type_path=type_path)
        query_dataset = self.get_dataset(tokenizer=victim_tokenizer, type_path=type_path)
        result_file_path = os.path.join(self.hparams.output_dir, result_path)
        os.makedirs(result_file_path, exist_ok=True)
        result_file_name = os.path.join(
            result_file_path, 
            f'global_epoch={self.hparams.global_epoch}_' + result_file
        )
        attack_file_name = os.path.join(
            result_file_path, 
            f'global_epoch={self.hparams.global_epoch}_' + attack_file
        )

        ppo_trainer = PPOTrainer(
            self.ppo_config, 
            self.model, 
            self.ref_model, 
            self.tokenizer, 
            dataset=dataset, 
            data_collator=dataset.collate_fn, 
        )
        
        query_ppo_trainer = PPOTrainer(
            self.ppo_config, 
            self.model, 
            self.ref_model, 
            self.tokenizer, 
            dataset=query_dataset, 
            data_collator=query_dataset.collate_fn, 
        )
        

        self.model.eval()
        victim_model = victim_model.to('cuda:0').eval()
        if self.train_type == 'backward': attr_model = attr_model.to('cuda:0').eval()
        metric_list = []
        # with open(result_file_name, 'w') as f:
        if True:
            if write_file: f = open(result_file_name, 'w')
            titles = ['original', 'label', 'original_label0_score', 'original_label1_score', 
            'generated', 'predict_label', 'generated_label0_score', 'generated_label1_score', ]
            #if write_file: f.write(','.join(titles) + '\n')
            ori_preds_list, preds_list, labels = [], [], []
            attr_preds_list = []
            for data, query_data in tqdm(zip(ppo_trainer.dataloader, query_ppo_trainer.dataloader)):
            # for data in tqdm(ppo_trainer.dataloader):
                query_tensors = [query for query in data["input_ids"]]
                query_lists = [ppo_trainer.generate(query, **self.model_generation_kwargs).squeeze() \
                                                                                for query in query_tensors]
                texts = [self.tokenizer.decode(query, skip_special_tokens=True) for query in query_lists]
                # regenerate batch data
                gen_data = [dict(src_texts=text, tgt_texts=label) for text, label in zip(texts, data['labels'])]
                gen_data = query_dataset.collate_fn(gen_data)

                keyword_not_pass = ['src_texts', 'tgt_texts', ]
                batch_input = {key: value for key, value in data.items() if key not in keyword_not_pass}
                gen_batch_input = {key: value for key, value in gen_data.items() if key not in keyword_not_pass}
                for key, value in gen_batch_input.items(): gen_batch_input[key] = value.to('cuda:0')
                
                # query victim model
                outputs = victim_model(**batch_input)
                gen_outputs = victim_model(**gen_batch_input)

                probs = outputs.logits.softmax(dim=1).tolist()
                gen_probs = gen_outputs.logits.softmax(dim=1).tolist()
                ori_preds = outputs.logits.softmax(dim=1).argmax(dim=1).tolist()
                preds = gen_outputs.logits.softmax(dim=1).argmax(dim=1).tolist()

                ori_preds_list += ori_preds 
                preds_list += preds 
                labels += [int(l) for l in data['tgt_texts']]

                # query attribute model
                if self.train_type == 'backward':
                    attr_outputs = attr_model(**gen_batch_input)
                    attr_probs = attr_outputs.logits.softmax(dim=1).tolist()
                    attr_preds = attr_outputs.logits.softmax(dim=1).argmax(dim=1).tolist()
                    attr_preds_list += attr_preds

                # write result 
                if write_file:
                    for src, tgt, [src_p0, src_p1], text, pred, [text_p0, text_p1] in \
                            zip(data['src_texts'], data['tgt_texts'], probs, texts, preds, gen_probs):
                        #text = text.replace("\n", " ")
                        #text = " ".join(text.split())
                        #f.write(f'"{src}","{tgt}", "{src_p0}", "{src_p1}","{text}", "{pred}", "{text_p0}", "{text_p1}"\n')
                        metric_list.append({
                            'original': src, 
                            'label': tgt, 
                            'original_label0_score': src_p0, 
                            'original_label1_score': src_p1, 
                            'generated': text, 
                            'predict_label': pred, 
                            'generated_label0_score': text_p0, 
                            'generated_label1_score':text_p1, 
                        })
            json.dump(metric_list, f, indent=4)

            #if write_file: f.close()

            # calculate acc
            ori_acc = len([1 for p, l in zip(ori_preds_list, labels) if p == l]) / len(labels)
            gen_acc = len([1 for p, l in zip(preds_list, labels) if p == l]) / len(labels)
            print(f'victim model accuracy on original data: {ori_acc}')
            print(f'victim model accuracy on adversarial data: {gen_acc}')
            if self.train_type == 'backward':
                attr_acc = len([1 for p in attr_preds_list if p == self.hparams.attribute_id]) / len(attr_preds_list)
                print(f'the ratio of the data generated by backward model predicted as the style{self.hparams.attribute_id}: {attr_acc}')

        


    def save_checkpoint(
        self, 
        epoch_num, 
        cls_dir='cls', 
        model_dir='model', 
        ref_model_dir='ref_model', 
        hparams_file='hparams.pkl', 
    ):
        model_save_path = os.path.join(self.hparams.output_dir, model_dir, f'epoch={epoch_num}')
        hparams_save_path = os.path.join(self.hparams.output_dir, hparams_file)
        os.makedirs(model_save_path, exist_ok=True)
        # torch.save(self.state_dict(), save_path) # would raise mutated error
        ''' should save separately instead of saving the whole pl module. '''
        ''' only save trained forward/backward model. '''
        self.tokenizer.save_pretrained(model_save_path)
        self.model.save_pretrained(model_save_path)
        pickle_save(argparse.Namespace(**self.hparams), hparams_save_path)

    @classmethod
    def load_checkpoint(
        cls, 
        model_args, 
        output_dir, 
        model_path, 
        dataset_path, 
        cls_dir='cls', 
        model_dir='model', 
        ref_model_dir='ref_model', 
        hparams_file='hparams.pkl', 
        train_type='forward',
        use_cls_type='victim', 
        dataset_type='victim', 
    ):
        hparams_load_path = os.path.join(model_path, output_dir, hparams_file)
        hparams = pickle_load(hparams_load_path)
        model_load_path = os.path.join(hparams.output_dir, model_dir)
        # global_epoch
        model_names = ['forward_model_name', 'backward_model_name', ] # reload only forward/backward model
        for name in model_names:
            value = getattr(model_args, name, None)
            setattr(hparams, name, value)
        return PPO(
            [hparams, ], 
            model_path, 
            dataset_path, 
            train_type=train_type,
            use_cls_type=use_cls_type, 
            dataset_type=dataset_type, 
        )
        

    def configure_callbacks(self):
        # Define your callbacks here
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath='./checkpoints',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
        return [checkpoint_callback]

    @staticmethod
    def add_model_specific_args():
        @dataclass
        class ModelArguments:
            forward_model_name: Optional[str] = field(
                default="lvwerra/gpt2-imdb", 
                metadata={"help": "the forward model name"}
            )
            forward_ref_model_name: Optional[str] = field(
                default="lvwerra/gpt2-imdb", 
                metadata={"help": "the forward reference model name"}
            )
            backward_model_name: Optional[str] = field(
                default="lvwerra/gpt2-imdb", 
                metadata={"help": "the backward model name"}
            )
            backward_ref_model_name: Optional[str] = field(
                default="lvwerra/gpt2-imdb", 
                metadata={"help": "the backward reference model name"}
            )
            victim_model_name: Optional[str] = field(
                default=None, 
                metadata={"help": "the victim model name"}
            )
            attribute_model_name: Optional[str] = field(
                default=None, 
                metadata={"help": "the attribute model name"}
            )
            output_dir: Optional[str] = field(
                default=None, 
                metadata={"help": "the ppo model store path. (unnecessary)"}
            )
            use_cls_pipeline: bool = field(
                default=False, 
                metadata={"help": "use pipeline for classification if set to true. "}
            )
            attribute_id: Optional[int] = field(
                default=None, 
                metadata={"help": "the style of the backward model. "}
            )
            model_save_period: Optional[int] = field(
                default=1, 
                metadata={"help": "the num of the periods of model save. "}
            )

        @dataclass
        class DataArguments:
            data_load_from_hf: bool = field(
                default=False, 
                metadata={"help": "load hf dataset if set to true. "}
            )
            victim_dataset_name: Optional[str] = field(
                default=None, 
                metadata={"help": "the victim dataset name"}
            )
            attribute_dataset_name: Optional[str] = field(
                default=None, 
                metadata={"help": "the attribute dataset name"}
            )
            n_train: Optional[int] = field(
                default=-1, 
                metadata={"help": "the number of samples for train, -1 means use all. "}
            )
            n_val: Optional[int] = field(
                default=-1, 
                metadata={"help": "the number of samples for train, -1 means use all. "}
            )
            n_test: Optional[int] = field(
                default=-1, 
                metadata={"help": "the number of samples for train, -1 means use all. "}
            )

        @dataclass
        class TrainingArguments:
            # train
            seed: Optional[int] = field(
                default=112, 
                metadata={"help": "the magic seed. "}
            )
            learning_rate: Optional[float] = field(
                default=1.41e-5, 
                metadata={"help": "the learning rate"}
            )
            mini_batch_size: Optional[int] = field(
                default=16, 
                metadata={"help": "the PPO minibatch size. "}
            )
            train_batch_size: Optional[int] = field(
                default=256, 
                metadata={"help": "the train batch size, should be divided by mini_batch_size. "}
            )
            gradient_accumulation_steps: Optional[int] = field(
                default=1, 
                metadata={"help": "the number of gradient accumulation steps"}
            )
            log_with: Optional[str] = field(
                default=None, 
                metadata={"help": "use 'wandb' to log with wandb"}
            )
            max_source_length: Optional[int] = field(
                default=20, 
                metadata={"help": "max source length for train. "}
            )
            min_source_length: Optional[int] = field(
                default=20, 
                metadata={"help": "min source length for train. "}
            )
            max_target_length: Optional[int] = field(
                default=20, 
                metadata={"help": "max target length for train. "}
            )
            min_target_length: Optional[int] = field(
                default=20, 
                metadata={"help": "min target length for train. "}
            )
            pad_to_max_len: bool = field(
                default=False, 
                metadata={"help": "true if pad to max length. "}
            )
            num_train_epochs: Optional[int] = field(
                default=5, 
                metadata={"help": "number of epoch for train. "}
            )
            # generation
            max_gen_target_length: Optional[int] = field(
                default=20, 
                metadata={"help": "max length for generation. "}
            )
            min_gen_target_length: Optional[int] = field(
                default=20, 
                metadata={"help": "min length for generation. "}
            )
            do_sample: bool = field(
                default=True, 
                metadata={"help": "whether to do sampling. "}
            )
            top_k: Optional[float] = field(
                default=0.0, 
                metadata={"help": "to get top k result. "}
            )
            top_p: Optional[float] = field(
                default=1.0, 
                metadata={"help": "to get top p result. "}
            )
            # other
            pad_token_id: Optional[int] = field(
                default=None, 
                metadata={"help": "tokenizer pad id. "}
            )
            # not used
            dropout: Optional[float] = field(
                default=0.0, 
                metadata={"help": "dropout rate. "}
            )
            label_smoothing: Optional[float] = field(
                default=0.0, 
                metadata={"help": "label smoothing rate. "}
            )
            weight_decay: Optional[float] = field(
                default=0.0, 
                metadata={"help": "weight decay rate. "}
            )
            early_stopping: Optional[bool] = field(
                default=False, 
                metadata={"help": "whether to early stop"}
            )
            target_kl: Optional[float] = field(
                default=0.1, 
                metadata={"help": "kl target for early stopping"}
            )

        return [ModelArguments, DataArguments, TrainingArguments]


def main():
    # get arguments
    parser = HfArgumentParser((*PPO.add_model_specific_args(), ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    model_path, dataset_path = './models', './data'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)

    if script_args.load_checkpoint_path is None:
        ppo_model = PPO(
            [model_args, data_args, training_args, ], 
            model_path, 
            dataset_path, 
            train_type=script_args.train_type,
            use_cls_type=script_args.use_cls_type, 
            dataset_type=script_args.dataset_type, 
        )
    else:
        print('load checkpoint from disk. ')
        ppo_model = PPO.load_checkpoint(
            model_args, 
            script_args.load_checkpoint_path, 
            model_path, 
            dataset_path, 
            train_type=script_args.train_type,
            use_cls_type=script_args.use_cls_type, 
            dataset_type=script_args.dataset_type, 
        )


    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.

    # We retrieve the dataloader by calling the `build_dataset` function.
    # dataset = build_dataset(config)

    # set seed before initializing value head for deterministic eval

    # Now let's build the model, the reference model, and the tokenizer.

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    """
    generation_kwargs = {
        "min_length": 20,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    """
    if script_args.do_train:
        ppo_model.train() # train without using pl trainer

    # ppo_model.save_checkpoint()

    # todo: do model inference, and store generate result to file
    
    if script_args.do_eval:
        ppo_model.generate()
    



# todos:
# load victim model, attribute classifier, two generators
# load dataset, victim: mrpc, attribute: yahoo
# two training process, forward and backward
# reward setting, forward: victim target score and attribute cls evenly score
# backward: attribute target score

if __name__ == '__main__':
    main()