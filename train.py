import argparse
import contextlib
import logging
import os
import random
import shutil
from functools import partial
from pathlib import Path

import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from peft import LoraConfig
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from utils import (
    collate_fn,
    compute_metrics,
    create_train_message,
    create_user_prompt,
    download_ct_image,
    download_ctrate_csv,
    extract_and_transform,
    preprocess_logits_for_metrics,
)

if __name__ == "__main__":
    # Set logging configuration
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    # Improve CUDA memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()

    # Set the number of samples used for training
    # The samples are downloaded directly from the train_fixed folder in the CT-RATE huggingface repo
    # On the LRZ mcml-hgx-a100-80x4 cluster, the maximum number of samples I could use were 1500 if DELETE_SAMPLE_AFTER_DOWNLOAD == True
    parser.add_argument(
        "--samples",
        default=1500,
        help="Number of samples used for training.",
        type=int,
    )

    # Set the number of 2D images created from each 3D CT scan
    # On the LRZ mcml-hgx-a100-80x4 cluster, the maximum number of 2D images I could use per sample were 11 if DELETE_SAMPLE_AFTER_DOWNLOAD == True
    # Must be set to at least 3
    parser.add_argument(
        "--images-per-sample",
        default=11,
        help="Number of 2D images created from each 3D CT scan.",
        type=int,
    )

    # Delete each downlaoded CT scan after converting it to IMAGES_PER_SAMPLE 2D images
    # Due to the 100GB storage limitations on the LRZ, this should be set to True unless developing with a small number of samples (<100)
    parser.add_argument(
        "--delete-sample-after-downloading",
        default=True,
        help="Delete each downlaoded CT scan after converting it to IMAGES_PER_SAMPLE 2D images.",
        type=bool,
    )

    # Set the folder where the finetuned model is saved
    parser.add_argument(
        "--model-folder",
        default="medgemma-finetuned",
        help="Output path for the finetuned medgemma model.",
        type=str,
    )

    # Set the optimizer
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="The optimizer used for training.",
        type=str,
    )

    # Set the number of training epochs
    parser.add_argument(
        "--epochs",
        default=50,
        help="Number of training epochs.",
        type=int,
    )

    # Set the learning rate
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        help="Learning rate.",
        type=float,
    )

    # Set the training batch size
    parser.add_argument(
        "--train-batch-size",
        default=1,
        help="Batch size per device during training.",
        type=int,
    )

    # Set the eval batch size
    parser.add_argument(
        "--eval-batch-size",
        default=1,
        help="Batch size per device during evaluation.",
        type=int,
    )

    # Set the gradient accumulation setps
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=4,
        help="Number of steps before performing a backward/update pass.",
        type=int,
    )

    # Set the lora alpha value
    parser.add_argument(
        "--lora-alpha",
        default=16,
        help="Scaling factor when applying lora weights to the weights.",
        type=int,
    )

    # Set the lora rank
    parser.add_argument(
        "--lora-rank",
        default=16,
        help="Lora attention dimension.",
        type=int,
    )

    # Set the lora dropout
    parser.add_argument(
        "--lora-dropout",
        default=0.05,
        help="The dropout probability for Lora layers.",
        type=float,
    )

    # Set the lora target modules
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help="The names of the modules to apply the adapter to.",
        type=str,
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Path to model checkpoint to continue training from a checkpoint.",
        type=str,
    )

    # Set the metric for early stopping
    parser.add_argument(
        "--metric-for-best-model",
        default="bleu",
        help="The metric used for early stopping.",
        type=str,
    )
    parser.add_argument(
        "--greater-is-better",
        default=True,
        help="Defines if the metric used for best model should be minimized or maximized.",
        type=bool,
    )

    args = parser.parse_args()
    if args.images_per_sample < 3:
        raise ValueError("'images_per_sample' must be at least 3.")

    # Set seed for reproducability
    # Randomness is used to only consider one scan_id and reporoduction per patient
    random.seed(42)

    # Create environmental variable with huggingface token
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        logger.error("No huggingface token found in environmental variables. See readme.")

    # Create folder for the 3D CT scans downloaded from huggingface
    directory = Path(__file__).parent
    data_folder = os.path.join(directory, "data")
    model_folder = os.path.join(directory, "models")
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # Delete existing model folder
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(os.path.join(model_folder, args.model_folder))

    # Create model folder and required subfolders
    os.makedirs(os.path.join(model_folder, args.model_folder), exist_ok=True)
    os.makedirs(os.path.join(model_folder, args.model_folder, "model"), exist_ok=True)
    os.makedirs(os.path.join(model_folder, args.model_folder, "processor"), exist_ok=True)

    # Download the label files
    reports = download_ctrate_csv(
        subfolder="radiology_text_reports",
        file="train_reports.csv",
        local_data_folder=data_folder,
    )

    # Download metadata
    metadata = download_ctrate_csv(
        subfolder="metadata",
        file="train_metadata.csv",
        local_data_folder=data_folder,
    )
    sex_mapping = {"M": "male", "F": "female"}

    data_dict = {"messages": [], "images": []}
    downloaded_samples = 0
    index = 0
    fs = HfFileSystem()
    patients = fs.glob("datasets/ibrahimhamamci/CT-RATE/dataset/train_fixed/*")
    while downloaded_samples < args.samples:
        patient_path = patients[index]
        index += 1
        # Download train_fixed samples from the ct rate dataset from huggingface
        try:
            img_array, filename = download_ct_image(patient_path=patient_path, local_data_folder=data_folder)
        except Exception:
            logger.warning(f"{index}: Downloading {patient_path.rsplit('/', 1)[1]} failed. Continue with next image.")
            continue
        else:
            logger.info(f"{index}: Download {patient_path.rsplit('/', 1)[1]} finished.")

        # Convert the 3D scan to 'images_per_sample' arrays of 2D images
        img_list, shapes = extract_and_transform(img_array=img_array, images_per_sample=args.images_per_sample)

        # Delete 3D image file to save disk space
        if args.delete_sample_after_downloading:
            shutil.rmtree(os.path.join(data_folder, *patient_path.rsplit("/", 3)[1:]))
            shutil.rmtree(os.path.join(data_folder, ".cache"))

        # Extract the patient's sex from the metadata file
        sex_metadata = metadata.loc[metadata["VolumeName"] == filename, "PatientSex"].values[0]
        sex = sex_mapping.get(sex_metadata, "unknown")

        # Extract the patient's age from the metadata file
        age_metadata = metadata.loc[metadata["VolumeName"] == filename, "PatientAge"].values[0]
        age = "unknown" if age_metadata.strip() == "" else int(age_metadata.rstrip("Y"))

        # Extract the manufacturer of the CT scanner from the metadata file
        manufacturer = metadata.loc[metadata["VolumeName"] == filename, "Manufacturer"].values[0]

        # Extract findings and impressions from the labels file
        findings = reports.loc[reports["VolumeName"] == filename, "Findings_EN"].values[0]
        impressions = reports.loc[reports["VolumeName"] == filename, "Impressions_EN"].values[0]

        # Append messages and images of the given sample
        data_dict["messages"].append(
            create_train_message(
                prompt=create_user_prompt(shapes=shapes, patient_sex=sex, patient_age=age, manufacturer=manufacturer),
                images_per_sample=args.images_per_sample,
                findings=findings,
                impression=impressions,
            )
        )
        data_dict["images"].append(img_list)

        # Iterate the number of successfully downloaded samples
        downloaded_samples += 1

    # Transform dataset into final input format
    logger.info("Create completed dataset.")
    data = DatasetDict({"train": Dataset.from_dict(data_dict)})

    # Split into train and validation sets
    data = data["train"].train_test_split(
        test_size=max(len(data["train"]) // 10, 1),
        shuffle=True,
        seed=42,
    )
    data["validation"] = data.pop("test")

    # Define model arguments
    logger.info("Load Model.")
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    # Load medgemma from huggingface
    MODEL_ID = "google/medgemma-4b-it"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    # Define lora arguments
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias="none",
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # Define training arguments
    train_args = SFTConfig(
        output_dir=os.path.join(
            model_folder, args.model_folder
        ),  # Directory and Hub repository id to save the model to
        num_train_epochs=args.epochs,  # Number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # Batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # Batch size per device during evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage
        optim=args.optim,  # Use fused AdamW optimizer for better performance
        logging_steps=100,  # Number of steps between logs
        save_strategy="epoch",  # Save checkpoint every epoch
        eval_strategy="epoch",  # Evaluate every eval_steps
        save_total_limit=1,  # Number of checkpoints that are saved on disk
        eval_steps=10,  # Number of steps between evaluations
        learning_rate=args.learning_rate,  # Learning rate based on QLoRA paper
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # Warmup ratio based on QLoRA paper
        lr_scheduler_type="linear",  # Use linear learning rate scheduler
        report_to="tensorboard",  # Report metrics to tensorboard
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={"skip_prepare_dataset": True},  # Skip default dataset preparation to preprocess manually
        remove_unused_columns=False,  # Columns are unused for training but needed for data collator
        label_names=["labels"],  # Input keys that correspond to the labels
        metric_for_best_model=args.metric_for_best_model,  # Metric used for early stopping
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        processing_class=processor,
        data_collator=partial(collate_fn, processor=processor),
        compute_metrics=partial(compute_metrics, processor=processor),  # type: ignore
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logger.info("Starting training.")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=os.path.join(model_folder, args.model_folder, args.resume_from_checkpoint))
    else:
        trainer.train()

    logger.info("Saving model.")
    trainer.save_model()

    logger.info("Finished execution.")
