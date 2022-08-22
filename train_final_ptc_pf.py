import json
import logging
import pandas as pd
import pyarrow.feather as feather
import sys
import torch
import transformers
from dataclasses import (
    dataclass, 
    field
)
from accelerate import Accelerator
from evaluate import load
from tqdm import (
    tqdm,
    trange
)
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    MBartAdapterModel,
    MBartConfig,
    MBartTokenizer,
    PfeifferConfig,
    set_seed
)
from typing import Optional


@dataclass(
    eq=False,
    frozen=True
)
class SpotArguments:
    task_name: Optional[str] = field()
    train_file: Optional[str] = field(default="train.feather")
    val_file: Optional[str] = field(default="val.feather")
    test_file: Optional[str] = field(default="test.feather")
    train_file_filtered: Optional[str] = field(default="train_filtered.parquet")
    val_file_filtered: Optional[str] = field(default="val_filtered.parquet")
    test_file_filtered: Optional[str] = field(default="test_filtered.parquet")   
    text_field: Optional[str] = field(default="headline")
    key_field: Optional[str] = field(default="campaignId")
    label_field: Optional[str] = field(default="labels")
    language_field: Optional[str] = field(default="lang")
    incidence_ignore_threshold: int = field(default=50)
    max_seq_length: int = field(default=128)
    train_batch_size: int = field(default=32)
    val_batch_size: int = field(default=32)
    test_batch_size: int = field(default=32)
    dataloader_num_workers: int = field(default=0)
    dataloader_prefetch_factor: int = field(default=2)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_drop_last: bool = field(default=True)
    dataloader_train_subset: Optional[int] = field(default=None)
    dataloader_val_subset: Optional[int] = field(default=None)
    dataloader_test_subset: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    model_name_or_path: str = field(default="facebook/mbart-large-cc25")
    cache_dir: Optional[str] = field(default=".")
    output_dir: Optional[str] = field(default=".")
    adapter_non_linearity: Optional[str] = field(default="relu")
    adapter_reduction_factor: Optional[float] = field(default=16.0)
    cross_adapter: bool = field(default=True)
    multihead_adapter: bool = field(default=True)
    epochs: int = field(default=8)
    output_adapter: bool = field(default=True)
    learning_rate: float = field(default=3e-4)
    weight_decay: float = field(default=3e-4)
    epsilon: float = field(default=1e-8)
    scheduler_warmup_prop: float = field(default=0.15)
    scheduler_num_cycles: float = field(default=0.5)
    clip_grad_threshold: float = field(default=2.0)
    log_level: Optional[str] = field(default="warning")
    seed: int = field(default=77)
    world_size: int = field(default=torch.cuda.device_count())
    sample_view_size: int = field(default=3)
        
    def get_log_level(self):
        if self.log_level == "debug":
            return logging.DEBUG
        elif self.log_level == "info":
            return logging.INFO
        elif self.log_level == "warning":
            return logging.WARNING
        elif self.log_level == "error":
            return logging.ERROR
        else:
            return logging.CRITICAL


class AdapterDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        file, 
        label2id,
        tokenizer_de_de,
        tokenizer_en_en,
        logger,
        max_length=128,
        text_field="asinProductTitle", 
        label_field="label",
        language_field="lang",
        sample=None
    ):
        self.label2id = label2id
        self.tokenizer_de_de = tokenizer_de_de
        self.tokenizer_en_en = tokenizer_en_en
        self.logger = logger
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field
        self.language_field = language_field
        self.frame = pd.read_parquet(
            file,
            engine="pyarrow",
            columns=[
                self.text_field, 
                self.label_field, 
                self.language_field
            ]
        )
        if sample is not None:
            if sample < self.frame.shape[0]:
                self.frame = self.frame[:sample]
            else:
                raise ValueError("Sample size larger than dataset")
        self.frame = self.frame.sample(frac=1.0)
        self.size = self.frame.shape[0]
        self.logger.info(f"Adapter dataset of size {self.size} initialized")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx > self.size:
            raise ValueError("Index out of bounds")
            
        label = self.label2id[self.frame.at[idx, self.label_field]]
        text = self.frame.at[idx, self.text_field]
        lang = self.frame.at[idx, self.language_field]
        if lang == "DE":
            processed_tokens = self.tokenizer_de_de.encode_plus(
                text,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True
            )
        elif lang == "EN":
            processed_tokens = self.tokenizer_en_en.encode_plus(
                text,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True
            )
        else:
            raise ValueError("Language not supported")
        return processed_tokens["input_ids"][0], processed_tokens["attention_mask"][0], label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(spot_args):
    accelerator = Accelerator(
        device_placement=True,
        split_batches=True,
        rng_types=["torch", "cuda", "generator"],
        step_scheduler_with_optimizer=True
    )
    

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(spot_args.get_log_level())
    transformers.utils.logging.set_verbosity(spot_args.get_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters: {spot_args}")
    logger.info(accelerator.state)
    set_seed(spot_args.seed)

    data_df_train = feather.read_feather(spot_args.train_file)
    logger.debug(f"Loaded train dataframe shape: {data_df_train.shape}")
    logger.debug(f"Sample train dataframe rows: {data_df_train.sample(frac=1.0).head(spot_args.sample_view_size)}")
    data_df_val = feather.read_feather(spot_args.val_file)
    logger.debug(f"Loaded val dataframe shape: {data_df_val.shape}")
    logger.debug(f"Sample val dataframe rows: {data_df_val.sample(frac=1.0).head(spot_args.sample_view_size)}")
    data_df_test = feather.read_feather(spot_args.test_file)
    logger.debug(f"Loaded test dataframe shape: {data_df_test.shape}")
    logger.debug(f"Sample test dataframe rows: {data_df_test.sample(frac=1.0).head(spot_args.sample_view_size)}")

    grouped_train_df = data_df_train.groupby([spot_args.label_field]).count()
    grouped_train_df = grouped_train_df[grouped_train_df[spot_args.key_field] > spot_args.incidence_ignore_threshold]
    subset_labels = list(grouped_train_df.index)
    subset_labels.sort()
    logger.debug(f"Sorted label set: {subset_labels}")
    num_labels = len(subset_labels)
    logger.debug(f"Number of labels: {num_labels}")

    id2label = {i : subset_labels[i] for i in range(num_labels)}
    logger.debug(f"ID-to-Label: {id2label}")
    label2id = {subset_labels[i] : i for i in range(num_labels)}
    logger.debug(f"Label-to-ID: {label2id}")

    if accelerator.is_main_process:
        data_df_train = data_df_train[data_df_train[spot_args.label_field].isin(subset_labels)]
        data_df_train = data_df_train.reset_index()
        logger.debug(f"Filtered train dataframe shape: {data_df_train.shape}")
        data_df_val = data_df_val[data_df_val[spot_args.label_field].isin(subset_labels)]
        data_df_val = data_df_val.reset_index()
        logger.debug(f"Filtered val dataframe shape: {data_df_val.shape}")
        data_df_test = data_df_test[data_df_test[spot_args.label_field].isin(subset_labels)]
        data_df_test = data_df_test.reset_index()
        logger.debug(f"Filtered test dataframe shape: {data_df_test.shape}")

        data_df_train.to_parquet(spot_args.train_file_filtered, index=False)
        data_df_val.to_parquet(spot_args.val_file_filtered, index=False)
        data_df_test.to_parquet(spot_args.test_file_filtered, index=False)
        logger.info("Filtered parquet files written")

    with accelerator.main_process_first():
        config = MBartConfig.from_pretrained(
            spot_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=spot_args.cache_dir,
        )
        tokenizer_de_de = MBartTokenizer.from_pretrained(
            spot_args.model_name_or_path,
            cache_dir=spot_args.cache_dir,
            use_fast=True,
            src_lang="de_DE",
            tgt_lang="de_DE"
        )
        tokenizer_en_en = MBartTokenizer.from_pretrained(
            spot_args.model_name_or_path,
            cache_dir=spot_args.cache_dir,
            use_fast=True,
            src_lang="en_XX",
            tgt_lang="en_XX"
        )
        model = MBartAdapterModel.from_pretrained(
            spot_args.model_name_or_path,
            config=config,
            cache_dir=spot_args.cache_dir,
            label2id=label2id,
            id2label=id2label
        )

    original_params = count_parameters(model)
    logger.debug(f"Original parameters: {original_params}")
    model.add_classification_head(
        spot_args.task_name,
        num_labels=num_labels,
        id2label=id2label,
    )
    original_params_with_cls = count_parameters(model)
    logger.debug(f"Original parameters with classifier head: {original_params_with_cls}")
    adapter_config = PfeifferConfig(
        cross_adapter=spot_args.cross_adapter,
        output_adapter=spot_args.output_adapter,
        mh_adapter=spot_args.output_adapter,
        non_linearity=spot_args.adapter_non_linearity,
        reduction_factor=spot_args.adapter_reduction_factor,
    )
    model.add_adapter(
        spot_args.task_name, 
        config=adapter_config
    )
    final_params = count_parameters(model)
    logger.debug(f"Final parameters: {final_params}")
    logger.info(f"Number of added parameters: {final_params - original_params_with_cls}")
    model.train_adapter([spot_args.task_name])
    model.set_active_adapters([spot_args.task_name])
    
    train_dataset = AdapterDataset(
        spot_args.train_file_filtered,
        label2id,
        tokenizer_de_de,
        tokenizer_en_en,
        logger,
        max_length=spot_args.max_seq_length,
        text_field=spot_args.text_field, 
        label_field=spot_args.label_field,
        language_field=spot_args.language_field,
        sample=spot_args.dataloader_train_subset
    )
    val_dataset = AdapterDataset(
        spot_args.val_file_filtered,
        label2id,
        tokenizer_de_de,
        tokenizer_en_en,
        logger,
        max_length=spot_args.max_seq_length,
        text_field=spot_args.text_field, 
        label_field=spot_args.label_field,
        language_field=spot_args.language_field,
        sample=spot_args.dataloader_val_subset
    )
    test_dataset = AdapterDataset(
        spot_args.test_file_filtered,
        label2id,
        tokenizer_de_de,
        tokenizer_en_en,
        logger,
        max_length=spot_args.max_seq_length,
        text_field=spot_args.text_field, 
        label_field=spot_args.label_field,
        language_field=spot_args.language_field,
        sample=spot_args.dataloader_test_subset
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=spot_args.train_batch_size,
        shuffle=True,
        pin_memory=spot_args.dataloader_pin_memory,
        num_workers=spot_args.dataloader_num_workers,
        prefetch_factor=spot_args.dataloader_prefetch_factor,
        drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=spot_args.val_batch_size,
        shuffle=False,
        pin_memory=spot_args.dataloader_pin_memory,
        num_workers=spot_args.dataloader_num_workers,
        prefetch_factor=spot_args.dataloader_prefetch_factor,
        drop_last=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=spot_args.test_batch_size,
        shuffle=False,
        pin_memory=spot_args.dataloader_pin_memory,
        num_workers=spot_args.dataloader_num_workers,
        prefetch_factor=spot_args.dataloader_prefetch_factor,
        drop_last=True
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": spot_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=spot_args.learning_rate,
        eps=spot_args.epsilon,
        correct_bias=True,
        no_deprecation_warning=True
    )
    num_warmup_steps = (len(train_dataloader.dataset) // spot_args.train_batch_size) * spot_args.epochs * spot_args.scheduler_warmup_prop
    num_training_steps = (len(train_dataloader.dataset) // (spot_args.train_batch_size * spot_args.world_size)) * spot_args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_cycles=spot_args.scheduler_num_cycles,
        num_training_steps=num_training_steps
    )
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        test_dataloader,
        scheduler
    )
    with accelerator.main_process_first():
        metric_accuracy = load("accuracy")
        metric_recall = load("recall")
        metric_precision = load("precision")
        metric_f1 = load("f1")

    accelerator.wait_for_everyone()
    summary_tracker = {}
    summary_tracker["parameters_added"] = final_params - original_params_with_cls
    for epoch_i in trange(spot_args.epochs, desc='Epoch Loop', leave=True, disable=not accelerator.is_main_process, position=0):
        model.train()
        total_train_loss = torch.zeros([1], device=accelerator.device)
        for input_ids, attention_mask, labels in tqdm(train_dataloader, desc='Train Loop', leave=False, disable=not accelerator.is_main_process, position=1):
            model.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            train_loss = out.loss
            total_train_loss+= train_loss.detach().mean()
            accelerator.backward(train_loss)
            accelerator.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=spot_args.clip_grad_threshold,
                norm_type=2.0,
            )
            optimizer.step()
            scheduler.step()
        total_train_loss = accelerator.reduce(total_train_loss, reduction="sum")

        model.eval()
        total_val_loss = torch.zeros([1], device=accelerator.device)
        for input_ids, attention_mask, labels in tqdm(val_dataloader, desc='Val Loop', leave=False, disable=not accelerator.is_main_process, position=1):
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = out.loss
                total_val_loss+= val_loss.detach().mean()
            predictions = out.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, labels))
            metric_accuracy.add_batch(predictions=predictions, references=references)
            metric_recall.add_batch(predictions=predictions, references=references)
            metric_precision.add_batch(predictions=predictions, references=references)
            metric_f1.add_batch(predictions=predictions, references=references)
        total_val_loss = accelerator.reduce(total_val_loss, reduction="sum")
            
        result_accuracy = metric_accuracy.compute()
        result_recall = metric_recall.compute(average="macro")
        result_precision = metric_precision.compute(average="macro")
        result_f1 = metric_f1.compute(average="macro")
        epoch_tracker = {}
        epoch_tracker["train_loss"] = total_train_loss.detach().item()
        epoch_tracker["val_loss"] = total_val_loss.detach().item()
        epoch_tracker["val_accuracy"] = result_accuracy["accuracy"]
        epoch_tracker["val_recall"] = result_recall["recall"]
        epoch_tracker["val_precision"] = result_precision["precision"]
        epoch_tracker["val_f1"] = result_f1["f1"]
        summary_tracker[f"epoch_{epoch_i}"] = epoch_tracker
        
        accelerator.save_state(f"{spot_args.output_dir}/epoch_{epoch_i}/accelerator/")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{spot_args.output_dir}/epoch_{epoch_i}/model/", 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            unwrapped_model.save_adapter(
                save_directory=f"{spot_args.output_dir}/epoch_{epoch_i}/adapter/",
                adapter_name=spot_args.task_name
            )
        accelerator.wait_for_everyone()

    for input_ids, attention_mask, labels in tqdm(test_dataloader, desc='Test Loop', leave=True, disable=not accelerator.is_main_process, position=1):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions = out.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, labels))
        metric_accuracy.add_batch(predictions=predictions, references=references)
        metric_recall.add_batch(predictions=predictions, references=references)
        metric_precision.add_batch(predictions=predictions, references=references)
        metric_f1.add_batch(predictions=predictions, references=references)

    result_accuracy = metric_accuracy.compute()
    result_recall = metric_recall.compute(average="macro")
    result_precision = metric_precision.compute(average="macro")
    result_f1 = metric_f1.compute(average="macro")
    test_tracker = {}
    test_tracker["test_accuracy"] = result_accuracy["accuracy"]
    test_tracker["test_recall"] = result_recall["recall"]
    test_tracker["test_precision"] = result_precision["precision"]
    test_tracker["test_f1"] = result_f1["f1"]
    summary_tracker["test"] = test_tracker
    if accelerator.is_main_process:
        logger.info(f"Run summary: {summary_tracker}")
        with open(f"{spot_args.output_dir}/summary_tracker.json", "w") as summary_writer:
            json.dump(
                summary_tracker, 
                summary_writer, 
                indent=4
            )
    return


if __name__ == "__main__":
    spot_args = SpotArguments(
        task_name="item_classification",
        text_field="title",
        key_field="id",
        label_field="label",
        incidence_ignore_threshold=50,
        max_seq_length=160,
        train_batch_size=128,
        val_batch_size=128,
        test_batch_size=128,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=8,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        dataloader_train_subset=None,
        dataloader_val_subset=None,
        dataloader_test_subset=None,
        epochs=12,
        cache_dir="/home/ec2-user/SageMaker/.huggingface/",
        output_dir="/home/ec2-user/SageMaker/Adapters/proj_name",
        cross_adapter=False,
        multihead_adapter=False,
        output_adapter=True,
        adapter_reduction_factor=4,
        learning_rate=3e-4,
        weight_decay=1e-3,
        log_level="debug",
        seed=4,
        sample_view_size=5
    )
    main(spot_args)
