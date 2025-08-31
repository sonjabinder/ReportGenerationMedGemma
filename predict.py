import argparse
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from dotenv import load_dotenv
from SimpleITK import GetArrayFromImage, ReadImage
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig

from utils import (
    collate_fn,
    create_inference_message,
    create_user_prompt,
    extract_and_transform,
)

if __name__ == "__main__":
    # Set logging configuration
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    parser = argparse.ArgumentParser()

    # input path

    # Set the number of 2D images created from each 3D CT scan
    # Must be set to at least 3
    parser.add_argument(
        "--images-per-sample",
        default=11,
        help="Number of 2D images created from each 3D CT scan.",
        type=int,
    )

    # Set the path of the pretrained model
    parser.add_argument(
        "--model-path",
        default="/opt/app/models/medgemma-finetuned",
        help="Path to a local directory where the model is saved.",
        type=str,
    )

    # Set an output file to save the created reports
    parser.add_argument(
        "--output-file",
        default="/output/results.json",
        help="Path to the output file.",
        type=str,
    )

    # Set an input path where the 3D scans are stored.
    parser.add_argument(
        "--input-path",
        default="/input",
        help="Path to a local directory to save the created reports.",
        type=str,
    )

    # Set the batch size.
    parser.add_argument(
        "--eval-batch-size",
        default=1,
        help="Path to a local directory to save the created reports.",
        type=int,
    )

    args = parser.parse_args()
    if args.images_per_sample < 3:
        raise ValueError("'images_per_sample' must be at least 3.")

    # Create environmental variable with huggingface token
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        logger.error("No huggingface token found in environmental variables. See readme.")

    data_dict = {"messages": [], "images": []}
    vols = sorted([p for p in Path(args.input_path).iterdir() if p.suffix.lower() in {".mha", ".nii", ".nii.gz"}])
    for v in vols:
        # Read input image
        itk = ReadImage(fileName=v)
        img_array = np.transpose(GetArrayFromImage(image=itk).astype("float32"), (2, 1, 0))

        # Convert the 3D scan to 'images_per_sample' arrays of 2D images
        img_list, shapes = extract_and_transform(img_array=img_array, images_per_sample=args.images_per_sample)

        # Append messages and images of the given sample
        data_dict["messages"].append(
            create_inference_message(
                prompt=create_user_prompt(
                    shapes=shapes, patient_sex="unknown", patient_age="unknown", manufacturer="unknown"
                ),
                images_per_sample=args.images_per_sample,
            )
        )
        data_dict["images"].append(img_list)
        logger.info(f"{v.name}: Successfully converted to 2D images.")

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

    # Predict all samples
    reports = []
    for v, batch in zip(vols, test_dataloader):
        # Try block to ensure an output for every sample
        try:
            batch = batch.to("cuda")

            # Get model predictions
            output = model.generate(**batch, max_new_tokens=1000)
            output_msg = processor.tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"{v.name} output: {output_msg}")

            # Extract findings and impressions from model output
            findings_match = re.search(r"findings:\s*(.*?)(?=\n\S+:|\Z)", output_msg, re.IGNORECASE | re.DOTALL)
            findings = findings_match.group(1).strip() if findings_match else "Missing."
            impressions_match = re.search(
                r"impression[s]?:\s*(.*?)(?=\n\S+:|\Z)", output_msg, re.IGNORECASE | re.DOTALL
            )
            impressions = impressions_match.group(1).strip() if impressions_match else "Missing."

            # Concatenate findings and impressions
            result_txt = f"Findings: {findings} Impressions: {impressions}"

            reports.append({"input_image_name": v.name.split(".")[0], "report": result_txt})

        except Exception:
            reports.append(
                {"input_image_name": v.name.split(".")[0], "report": "Impressions: Missing. Findings: Missing."}
            )
            logger.info(f"{v.name} failed.")

    result = {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": reports,
        "version": {"major": 1, "minor": 0},
    }
    logger.info(result)

    with Path(args.output_file).open("w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved output file: {args.output_file}")
