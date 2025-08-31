"""Helper methods for finetuning MedGemma for medical report generation of 3D CT scans."""

import random
from typing import Any, Literal

import evaluate
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image
from SimpleITK import GetArrayFromImage, ReadImage
from transformers import EvalPrediction
from transformers.models.auto.image_processing_auto import AutoImageProcessor

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")


def collate_fn(examples: list[dict[str, Any]], processor: AutoImageProcessor) -> torch.Tensor:
    """Create a batch of correctly formatted samples.

    Args:
        examples (list[dict[str, Any]]): List of samples returned by the __getitem__ method of a Dataset class.
        processor (AutoImageProcessor): Image processor used for training.

    Returns:
        torch.Tensor: Batch in the correct input format.

    """
    texts = []
    images = []
    for example in examples:
        images.append([img.convert("RGB") for img in example["images"]])
        texts.append(
            processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False).strip()
        )

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, with the padding and image tokens masked in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])]

    # Mask tokens that are not used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels

    return batch


def create_user_prompt(
    shapes: tuple[int, int, int], patient_sex: str, patient_age: int | Literal["unknown"], manufacturer: str
) -> str:
    """Create a prompt for medical report generation of chest CT scan.

    Args:
        shapes (tuple[int, int, int]): x, y, and z axis length of the axial view after resizing the image.
        patient_sex (str): Sex of the patient.
        patient_age (int | Literal["unknown"]): Age of the patient in years.
        manufacturer (str): Manfucaturer of the CT scanner.

    Returns:
        str: Prompt.

    """
    return f"""Write a radiology report for this CT-Scan. Correctly describe normal findings and pathologies. Use standard chest CT terminology in your report. Cover findings and impression.

    The patients sex is {patient_sex} and they are {patient_age} years old. {manufacturer} is the manufacturer of the CT scanner.

    Each image consists of three sections from different perspectives. 
    The top left section (size {shapes[0]} times {shapes[2]}) shows the sagittal view, the bottom left section (size {shapes[0]} times {shapes[1]}) the axial view, and the bottom right section (size {shapes[2]} times {shapes[1]}) the coronal view.
    """


def v_shaped_steps(min_val: int, max_val: int, n: int, centering: float = 0.3) -> list[int]:
    """Generate a list of 'n' values between 'min_val' and 'max_val' with V-shaped step differences.

    The steps between the values are largest at the ends and smallest in the middle.
    The 'centering' parameter controls how strong this V-shape is.

    Args:
        min_val (int): The starting value of the sequence.
        max_val (int): The ending value of the sequence.
        n (int): Number of values to generate (including 'min_val' and 'max_val').
        centering (float): A value between 0 and 1 that controls the strength of the V-shape:
            - 0.0 = strong centering (small steps in the middle)
            - 1.0 = almost uniform steps

    Returns:
        list[int]: A list of 'n' integer values from 'min_val' to 'max_val' with V-shaped step sizes.

    """
    steps = n - 1  # Number of intervals between values
    midpoint = steps // 2

    # Calculate maximum and minimum step sizes
    max_step = (max_val - min_val) / (midpoint if midpoint > 0 else 1)
    min_step = max_step * max(0.01, min(centering, 1))

    # Generate left and right parts of the V-shaped differences
    if steps % 2 == 0:
        left = np.linspace(max_step, min_step, midpoint)
        right = left[::-1]
        differences = np.concatenate((left, right))
    else:
        left = np.linspace(max_step, min_step, midpoint)
        right = left[::-1]
        middle = np.array([min_step])
        differences = np.concatenate((left, middle, right))

    # Scale the differences so they sum exactly to (max_val - min_val)
    total_diff = differences.sum()
    differences = differences * ((max_val - min_val) / total_diff)

    # Round to integers
    differences = np.round(differences).astype(int)

    # Correct rounding error by adjusting the largest step
    diff_sum = differences.sum()
    target_diff = max_val - min_val
    correction = target_diff - diff_sum
    if correction != 0:
        max_idx = np.argmax(differences)
        differences[max_idx] += correction

    # Build the list of values from min_val using the step differences
    v_shaped_values = [min_val]
    for d in differences:
        v_shaped_values.append(v_shaped_values[-1] + d)

    return v_shaped_values


def download_ct_image(patient_path: str, local_data_folder: str) -> tuple[np.ndarray, str]:
    """Download a 3D ct scan from the CT-RATE dataset.

    If a patient was scanned at multiple points of time or multiple reconstructions of the same scan are available, there is only one random scan of this patient picked.

    Args:
        patient_path (str): Huggingface path to a patient directory in the CT-RATE dataset.
        local_data_folder (str): Local directory to save the 3D ct scan.

    Raises:
        Exception: If an exception is raised during download or when loading the image file.

    Returns:
        tuple[np.ndarray, str]: Tuple with an array of the pixel values of the 3D scan and the name of the file.

    """
    repo_id = "ibrahimhamamci/CT-RATE"
    fs = HfFileSystem()

    patient_scans = fs.ls(patient_path, detail=False)
    patient_scan = patient_scans[random.randint(0, len(patient_scans) - 1)]

    reconstructions = fs.ls(patient_scan, detail=False)  # type: ignore
    reconstruction = reconstructions[random.randint(0, len(reconstructions) - 1)]

    hf_path = reconstruction.split("/", 3)[-1]  # type: ignore
    hf_subfolder, filename = hf_path.rsplit("/", 1)
    local_filepath = f"{local_data_folder}/{hf_path}"

    try:
        itk = ReadImage(fileName=local_filepath)
    except RuntimeError:
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                subfolder=hf_subfolder,
                filename=filename,
                local_dir=local_data_folder,
            )
        except Exception as e:
            raise e

        try:
            itk = ReadImage(fileName=local_filepath)
        except Exception as e:
            raise e
    except Exception as e:
        raise e

    # Convert itk image to numpy array and swap axis
    img_array = np.transpose(GetArrayFromImage(image=itk).astype("float32"), (2, 1, 0))

    return (img_array, filename)


def download_ctrate_csv(subfolder: str, file: str, local_data_folder: str) -> pd.DataFrame:
    """Download and open a csv file from the CT-RATE repository.

    Args:
        subfolder (str): Subfolder in the CT-RATE repository.
        file (str): Name of the to be downloaded file.
        local_data_folder (str): Local directory to save the 3D ct scan.

    Returns:
        pd.DataFrame: Contents of the csv file.

    """
    repo_id = "ibrahimhamamci/CT-RATE"
    subfolder = f"dataset/{subfolder}"
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        subfolder=subfolder,
        filename=file,
        local_dir=local_data_folder,
    )
    return pd.read_csv(f"{local_data_folder}/{subfolder}/{file}")


def extract_and_transform(img_array: np.ndarray, images_per_sample: int) -> tuple[list, tuple[int, int, int]]:
    """Transform a 3D iamge array into 'images_per_sample' 2D arrays.

    Each 2D array consists of the axial, coronal, and sagittal view at a specific point of the 3D array.
    There are fewer 2D arrays taken from the ends and more from the middle of the 3D array as this part contains most of the information.

    Args:
        img_array (np.ndarray): 3D array of the image.
        images_per_sample (int): Number of 2D images created from the 3D image. Must be at least 3.

    Returns:
        tuple[list, tuple[int, int, int]]:
            - List of 2D arrays
            - Tuple consisting of the number of pixels in each 2D array of the x-axis of the axial view, x-axis of the sagittal view, and x-axis of the coronal view.

    """
    # Normalize image
    img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

    # Assert the axial view is quadratic, if not then pad it
    if img_normalized.shape[0] < img_normalized.shape[1]:
        padding_total = img_normalized.shape[1] - img_normalized.shape[0]
        pad_before = padding_total // 2
        pad_after = padding_total - pad_before
        img_normalized = np.pad(
            img_normalized, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant", constant_values=0
        )
    elif img_normalized.shape[0] > img_normalized.shape[1]:
        padding_total = img_normalized.shape[0] - img_normalized.shape[1]
        pad_before = padding_total // 2
        pad_after = padding_total - pad_before
        img_normalized = np.pad(
            img_normalized, ((0, 0), (pad_before, pad_after), (0, 0)), mode="constant", constant_values=0
        )

    # Determine shape of x, y, and z axis
    shape_0, shape_1, shape_2 = img_normalized.shape

    # Calculate the x, y, and z values where a 2D image is extracted
    steps_0 = v_shaped_steps(0, shape_0 - 1, images_per_sample)
    steps_1 = v_shaped_steps(0, shape_1 - 1, images_per_sample)
    steps_2 = v_shaped_steps(0, shape_2 - 1, images_per_sample)

    # Create a list of 'images_per_sample' 2D images, resized to 896x896 pixels
    img_list = []
    for step_0, step_1, step_2 in zip(steps_0, steps_1, steps_2):
        img_slice_0 = img_normalized[step_0, :, :]
        img_slice_1 = np.rot90(img_normalized[:, step_1, :], k=1)
        img_slice_2 = np.rot90(img_normalized[:, :, step_2], k=3)

        # Create padding for the remaining image
        img_slice_3 = np.zeros((shape_2, shape_2), dtype=np.uint8)

        # Combine the four image parts
        top_row = np.hstack((img_slice_1, img_slice_3))
        bottom_row = np.hstack((img_slice_2, img_slice_0))
        final_image = np.vstack((top_row, bottom_row))

        # Convert to PIL and resize to 896 x 896
        img_pil = Image.fromarray(final_image, mode="L")
        img_resize = img_pil.resize((896, 896), Image.Resampling.LANCZOS)

        img_list.append(img_resize)

    return img_list, (shape_0, shape_1, shape_2)


def create_train_message(
    prompt: str, images_per_sample: int, findings: str, impression: str
) -> list[dict[str, object]]:
    """Create a formatted model input for multimodal models.

    Args:
        prompt (str): Prompt message.
        images_per_sample (int): Number of 2D images for each sample.
        findings (str): Ground truth findings of the sample.
        impression (str): Ground truth impression of the sample.

    Returns:
        list[dict[str, object]]: Formatted model input.

    """
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert radiologist."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image"}] * images_per_sample,
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Findings:\n{findings}\n\nImpression:\n{impression}"},
            ],
        },
    ]
    return messages


def create_inference_message(prompt: str, images_per_sample: int) -> list[dict[str, object]]:
    """Create a formatted model input for multimodal models.

    Args:
        prompt (str): Prompt message.
        images_per_sample (int): Number of 2D images for each sample.

    Returns:
        list[dict[str, object]]: Formatted model input.

    """
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert radiologist."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image"}] * images_per_sample,
            ],
        },
    ]


def preprocess_logits_for_metrics(logits: tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Preprocess model logits for metric computation by converting them to predicted token IDs.

    This function is used to extract predictions from raw logits during evaluation. It applies
    'argmax' over the vocabulary dimension to obtain predicted token IDs for each position
    in the sequence.

    Args:
        logits (tuple[torch.Tensor, torch.Tensor]): A tuple containing the logits tensor of shape (batch_size, sequence_length, vocab_size).
        labels (Tensor): The ground truth labels tensor of shape (batch_size, sequence_length), unused.

    Returns:
        torch.Tensor: The predicted token IDs of shape (batch_size, sequence_length).

    """
    return torch.argmax(logits[0], dim=-1)


def compute_metrics(pred: EvalPrediction, processor: AutoImageProcessor) -> dict[str, float]:
    """Compute evaluation metrics (BLEU, ROUGE, METEOR) for model predictions vs. labels.

    This function decodes predictions and labels using the tokenizer from a processor, handles padding and special tokens, and computes standard NLP generation metrics.

    Args:
        pred (EvalPrediction): Model output scores and ground truth token ids.
        processor (AutoProcessor): Processor which is used to decode the input_ids

    Returns:
        dict[str, float]: Dictionariy with computed metrics.

    """
    preds, labels = pred

    # Cast the predictions and labels into correct format
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Replace -100 (ignore index) with pad token id for decoding
    labels[labels == -100] = processor.tokenizer.pad_token_id
    preds[preds == -100] = processor.tokenizer.pad_token_id

    # Convert tensors to lists
    preds_list = preds.tolist()
    labels_list = labels.tolist()

    # Decode token IDs to text
    decoded_preds = processor.tokenizer.batch_decode(preds_list, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels_list, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]

    # Compute metrics
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu["bleu"],  # type: ignore
        "bleu1": bleu["precisions"][0],  # type: ignore
        "bleu2": bleu["precisions"][1],  # type: ignore
        "bleu3": bleu["precisions"][2],  # type: ignore
        "bleu4": bleu["precisions"][3],  # type: ignore
        **rouge,  # type: ignore
        **meteor,  # type: ignore
    }
