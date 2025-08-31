import argparse
import contextlib
import logging
import os
import random
import re
import shutil
from functools import partial
from pathlib import Path

import torch
from datasets.arrow_dataset import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig

from utils import (
    collate_fn,
    compute_metrics,
    create_inference_message,
    create_user_prompt,
    download_ct_image,
    download_ctrate_csv,
    extract_and_transform,
)

if __name__ == "__main__":
    # Set logging configuration
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    parser = argparse.ArgumentParser()

    # Set the number of validation samples downloaded from the of ctrate dataset
    parser.add_argument(
        "--samples",
        default=1000,
        help="Number of samples downloaded from the CT-Rate dataset and subsequently predicted.",
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

    # Set the eval batch size
    parser.add_argument(
        "--eval-batch-size",
        default=1,
        help="Batch size per device during evaluation.",
        type=int,
    )

    # Set the path of the pretrained model
    parser.add_argument(
        "--model-path",
        default="./ehealth/models/medgemma-finetuned/checkpoint-4056/",
        help="Path to a local directory where the model is saved.",
        type=str,
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

    # Download the label files
    reports = download_ctrate_csv(
        subfolder="radiology_text_reports",
        file="validation_reports.csv",
        local_data_folder=data_folder,
    )

    # Download metadata
    metadata = download_ctrate_csv(
        subfolder="metadata",
        file="validation_metadata.csv",
        local_data_folder=data_folder,
    )
    sex_mapping = {"M": "male", "F": "female"}

    data_dict = {"messages": [], "images": []}
    filenames = []
    downloaded_samples = 0
    index = 0
    fs = HfFileSystem()
    patients = fs.glob("datasets/ibrahimhamamci/CT-RATE/dataset/valid_fixed/*")
    while downloaded_samples < args.samples:
        patient_path = patients[index]
        index += 1
        # Download valid_fixed samples from the ct rate dataset from huggingface
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
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(os.path.join(data_folder, ".cache"))

        # Extract the patient's sex from the metadata file
        sex_metadata = metadata.loc[metadata["VolumeName"] == filename, "PatientSex"].values[0]
        sex = sex_mapping.get(sex_metadata, "unknown")

        # Extract the patient's age from the metadata file
        age_metadata = metadata.loc[metadata["VolumeName"] == filename, "PatientAge"].values[0]
        age = "unknown" if age_metadata.strip() == "" else int(age_metadata.rstrip("Y"))

        # Extract the manufacturer of the CT scanner from the metadata file
        manufacturer = metadata.loc[metadata["VolumeName"] == filename, "Manufacturer"].values[0]

        # Append messages and images of the given sample
        data_dict["messages"].append(
            create_inference_message(
                prompt=create_user_prompt(shapes=shapes, patient_sex=sex, patient_age=age, manufacturer=manufacturer),
                images_per_sample=args.images_per_sample,
            )
        )
        data_dict["images"].append(img_list)
        filenames.append(filename)

        # Iterate the number of successfully downloaded samples
        downloaded_samples += 1

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

    # Load finetuned medgemma
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "right"

    # Transform dataset into final input format
    logger.info("Create completed dataset.")
    test_dataloader = DataLoader(
        dataset=Dataset.from_dict(data_dict),  # type: ignore
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor),
        drop_last=False,
    )

    # Predict all samples and calculate the metrics
    all_metrics = []
    for index, (batch, filename) in enumerate(zip(test_dataloader, filenames)):
        batch = batch.to("cuda")

        # Get model predictions
        output = model.generate(**batch, max_new_tokens=1000)
        output_msg = processor.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract findings and impressions from model output
        findings_match = re.search(r"findings:\s*(.*?)(?=\n\S+:|\Z)", output_msg, re.IGNORECASE | re.DOTALL)
        findings = findings_match.group(1).strip() if findings_match else "Missing."
        impressions_match = re.search(r"impression[s]?:\s*(.*?)(?=\n\S+:|\Z)", output_msg, re.IGNORECASE | re.DOTALL)
        impressions = impressions_match.group(1).strip() if impressions_match else "Missing."

        # Concatenate findings and impressions
        result_txt = f"Findings: {findings} Impressions: {impressions}".replace("<end_of_turn>", "")

        # Extract ground truth findings and impressions
        findings_label = reports.loc[reports["VolumeName"] == filename, "Findings_EN"].values[0]  # type: ignore
        impressions_label = reports.loc[reports["VolumeName"] == filename, "Impressions_EN"].values[0]  # type: ignore
        label_txt = f"Findings: {findings_label} Impressions: {impressions_label}"

        # Tokenize ground truth and predicted findings and labels
        labels_tok = processor(text=label_txt, return_tensors="pt", padding=True)["input_ids"]
        result_tok = processor(text=result_txt, return_tensors="pt", padding=True)["input_ids"]

        # Compute relevant metrics
        metrics = compute_metrics(pred=(result_tok, labels_tok), processor=processor)  # type: ignore
        metrics = {k: float(v) for k, v in metrics.items()}
        all_metrics.append(metrics)

        # Print findings, labels and calculated metrics
        logger.info(f"File: {filename}")
        logger.info(f"Ground truth: {label_txt}")
        logger.info(f"Model prediction: {result_txt}")
        logger.info(f"Raw model output: {output_msg}")
        logger.info(metrics)
        logger.info("\n\n\n-----\n\n\n")

    # Print means all relevant metrics
    logger.info("Mean metrics across all samples")
    logger.info(f"BLEU-1: {sum(d['bleu1'] for d in all_metrics) / len(all_metrics)}")
    logger.info(f"BLEU-2: {sum(d['bleu2'] for d in all_metrics) / len(all_metrics)}")
    logger.info(f"BLEU-3: {sum(d['bleu3'] for d in all_metrics) / len(all_metrics)}")
    logger.info(f"BLEU-4: {sum(d['bleu4'] for d in all_metrics) / len(all_metrics)}")
    logger.info(f"RougeL: {sum(d['rougeL'] for d in all_metrics) / len(all_metrics)}")
    logger.info(f"Meteor: {sum(d['meteor'] for d in all_metrics) / len(all_metrics)}")

    logger.info("Finished execution.")
