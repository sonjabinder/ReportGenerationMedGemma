# Radiology Report Generation of CT Chest Scans - VLM3D Challenge 2025

The aim of this project is to generate a free‑text radiology report from a 3D chest CT volume. The radiology report:
- Correctly describes normal findings and pathologies
- Uses standard chest CT terminology
- Covers findings and impression

Radiologists spend considerable time dictating comprehensive reports for chest CT scans. Automating this step can:
- Speed up diagnostic workflows
- Reduce variability between readers
- Improve patient care by enabling rapid triage

For further challenge details, see https://reportgen.vlm3dchallenge.com

For further details on the dataset, see https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

## Architecure

This project is based on the **MedGemma** architecture, a specialized extension of the Gemma model family designed for medical applications. Explicitly, the large multi model version is used. 

For further architecture details, see https://deepmind.google/models/gemma/medgemma/


## Setup

1. Clone the repository:
  ```bash
  git clone https://github.com/sonjabinder/ReportGenerationMedGemma.git
  cd ReportGenerationMedGemma
  ```

2. Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/Mac
  .venv\Scripts\activate     # Windows
  ```

3. Install dependencies:
  ```bash
  python3 -m pip install -r requirements.txt  # Linux/Mac
  py -m pip install -r requirements.txt       # Windows
  ```

4. Create a file `.env` and save your huggingface token with the required permissions for both MedGemma as well as the CT-RATE dataset in it with the following format 
  ```bash
  HF_TOKEN=<YOUR-HF-TOKEN>
  ```

## Project Structure

- **train.py**  
  Finetunes MedGemma for radiology report generation.
  
- **eval.py**  
  Computes relevant metrics using the finetuned model and the validation split of the CT-RATE dataset.

- **predict.py**  
  Generates radiology reports using CT scans from a local filesystem.

- **utils.py**  
  Contains helper methods used for training, evaluation, and prediction.

- **requirements.txt**  
  List of python dependencies required to run the project.

- **.env**  
  Must be created manually. Saves the token for huggingface permissions. 

- **Dockerfile**  
  Defines base image to containerize model.

- **docker/build.sh**  
  Helper to build the docker image locally.
  
- **docker/export.sh**  
  Exports the built image as a .tar file for transfer.
  
- **docker/test.sh**  
  Runs a prediction of a single sample in docker container to ensure correct functionality.

- **lrz/train.sbatch**  
  Submits a training job on the LRZ AI cluster.
  
- **lrz/evaluate.sbatch**  
  Submits an evaluation job on the LRZ AI cluster.
  
- **lrz/create_env.sbatch**  
  Sets up the environment on LRZ.
  

## Usage

Note: The model was trained using an Nvidia A100 GPU with 80GB of memory. With less memory the training parameters need to be adjusted accordingly.

Train the model using the CT-RATE dataset:
```bash
python train.py [--samples 1500] [--images-per-sample 11] [--delete-sample-after-downloading] \
	[--model-folder "medgemma-finetuned"] [--optim "adamw_torch_fused"] [--epochs 20] [--learning-rate 0.00001] \
	[--train-batch-size 1] [--eval-batch-size 1] [--gradient-accumulation-steps 4] [--lora-alpha 16] \
	[--lora-rank 16] [--lora-dropout 0.05] [--target-modules "all-linear"] [--resume-from-checkpoint ""] \
	[--metric-for-best-model "bleu"] [--greater-is-better True] 
```

Test the model with validation samples from the CT-RATE dataset:
```bash
python eval.py [--samples 100] [--images-per-sample 11] [--delete-sample-after-downloading] \
	[--eval-batch-size 1] [--model-path "models/medgemma-finetuned/model"] \
	[--processor-path "models/medgemma-finetuned/processor"]
```

Predict unseen test samples from a local directory:
```bash
python predict.py [--images-per-sample 11] [--model-path "/opt/app/models/medgemma-finetuned/model"] \
	[--processor-path "/opt/app/models/medgemma-finetuned/processor"] [--output-file "/output/results.json"] \
	[--input-path "/input"] [--eval-batch-size 1]
```


## Docker

After training, the finetuned model can be containerized with the `docker/build.sh` script. The docker container uses the `predict.py` script as entrypoint. The `docker/export.sh` script saves the docker image in an archive for submission in the VLM3D challenge. `docker/test.sh` creates a new docker volume and mounts the exemplatory CT scan from the `docker/test` folder as an input image. The resulting report can be found in the `results.json` in the docker volume. This is used to test the docker container for correct functionality before submission on the VLM3D challenge.


## 3D-Image Example

<img width = "400" src="https://github.com/user-attachments/assets/6693b8be-6f44-43e1-8ebc-6de85009c34b" />


## Transform 3D-Image to ``n``x 2D-Images

At `N` points in the 3D-Image, the sagittal, coronal, and axial view is extracted. For each of the `N` points, the sagittal, coronal, and axial views are subsequently combined to yield `N` input images.

<img width="898" height="439" src="https://github.com/user-attachments/assets/7f6d3d75-9e0c-4919-a660-32706cb7ca9c" />


## Results

Across 1000 unseen CT scans from the CT-RATE dataset the following metrics are achieved with the finetuned model.

<table>
<tr>
<td><b style="font-size:30px">BLEU-1</b></td>
<td><b style="font-size:30px">BLEU-2</b></td>
<td><b style="font-size:30px">BLEU-3</b></td>
<td><b style="font-size:30px">BLEU-4</b></td>
<td><b style="font-size:30px">RougeL</b></td>
<td><b style="font-size:30px">Meteor</b></td>
</tr>
<tr>
<td>0.6494</td>
<td>0.3460</td>
<td>0.2046</td>
<td>0.1364</td>
<td>0.3140</td>
<td>0.3469</td>
</tr>
</table>

<details>
<summary>Example Report 1</summary>

On the left hand side, the provided report from the dataset is shown. On the right hand side, the predicted model output is shown. The differences between those two reports for the same CT scan are highlighted. Below the reports, the metrics associated with the genereated report are given.

<table>
<tr>
<td><b style="font-size:30px">Ground Truth</b></td>
<td><b style="font-size:30px">Model Output</b></td>
</tr>
<tr>
<td>

Findings: Trachea and both main bronchi are open. <br/>
**There is** no **obstructive** pathology in the trachea and both main bronchi. <br/>
**There are several millimetric nonspecific nodules in the right lung.** <br/>
No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions. No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No enlarged lymph nodes in pathological dimensions were detected. In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. 
The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections. <br/>
<br/>
Impressions: **Several millimetric nonspecific nodules in the right lung**

</td>
<td>

Findings: Trachea and both main bronchi are open. <br/>
No **occlusive** pathology **was detected** in the trachea and both main bronchi. <br/>
<br/><br/>
No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions. No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No enlarged lymph nodes in pathological dimensions were detected. In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections. <br/>
<br/>
Impressions: **Findings within normal limits**<br/>
<br/>

</td>
</tr>
</table>

<table>
<tr>
<td><b style="font-size:30px">BLEU-1</b></td>
<td><b style="font-size:30px">BLEU-2</b></td>
<td><b style="font-size:30px">BLEU-3</b></td>
<td><b style="font-size:30px">BLEU-4</b></td>
<td><b style="font-size:30px">RougeL</b></td>
<td><b style="font-size:30px">Meteor</b></td>
</tr>
<tr>
<td>0.6317</td>
<td>0.5987</td>
<td>0.5911</td>
<td>0.5833</td>
<td>0.7338</td>
<td>0.8779</td>
</tr>
</table>

</details>


<details>
<summary>Example Report 2</summary>

On the left hand side, the provided report from the dataset is shown. On the right hand side, the predicted model output is shown. The differences between those two reports for the same CT scan are highlighted. Below the reports, the metrics associated with the genereated report are given.

<table>
<tr>
<td><b style="font-size:30px">Ground Truth</b></td>
<td><b style="font-size:30px">Model Output</b></td>
</tr>
<tr>
<td>

Findings: <br/>
Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. <br/>
**Central and peripheral consolidations and ground glass areas are observed in the middle lobe and lower lobe of the right lung. There are also small areas of ground glass and nodular-shaped consolidations in the left lung.** <br/>
The described **findings** were **evaluated** in favor of viral pneumonia. <br/>
**Findings described especially in the right lung are frequently encountered findings in Covid-19 pneumonia.** <br/>
No mass was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. <br/>
No enlarged lymph nodes **in pathological size and appearance** were detected in the mediastinum and hilar regions. <br/>
No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. <br/>
No **pathologically** enlarged lymph nodes were **observed**. <br/><br/>
In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections. <br/>
<br/>
Impressions: Findings **evaluated in favor of** viral pneumonia in both lungs<br/>

</td>
<td>

Findings: <br/> 
Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. <br/>
**Peripheral and centrally located ground glass areas are observed in the upper and lower lobes of both lungs and the middle lobe of the right lung.** <br/><br/>
The described **manifestations** were **first voted** in favor of viral pneumonia. <br/>
**These findings are frequently observed in Covid-19 pneumonia.** <br/>
No mass was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. <br/>
No **pathologically** enlarged lymph nodes were detected in the mediastinum and hilar regions. <br/>
No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. <br/>
No enlarged lymph nodes **in pathological dimensions** were **detected**. <br/>
In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections. <br/>
<br/>
Impressions: Findings **consistent with** viral pneumonia in both lungs<br/>

</td>
</tr>
</table>


<table>
<tr>
<td><b style="font-size:30px">BLEU-1</b></td>
<td><b style="font-size:30px">BLEU-2</b></td>
<td><b style="font-size:30px">BLEU-3</b></td>
<td><b style="font-size:30px">BLEU-4</b></td>
<td><b style="font-size:30px">RougeL</b></td>
<td><b style="font-size:30px">Meteor</b></td>
</tr>
<tr>
<td>0.6731</td>
<td>0.5978</td>
<td>0.5663</td>
<td>0.5346</td>
<td>0.7175</td>
<td>0.8711</td>
</tr>
</table>

</details>

