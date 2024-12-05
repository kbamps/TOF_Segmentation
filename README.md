# Quick Start Guide for TOF Segmentation with Docker
## Prerequisites:
Ensure Docker is installed and properly configured on your system.

## Step 1: Build the Docker Image
Run the following command to build the Docker image for TOF segmentation:

```bash
docker build -t tofsegmentation:latest .
```
## Step 2: Run the Docker Container
Execute the container using the command below, adjusting the paths as needed for your environment:

```bash
docker run --gpus all -v /path/to/input/data:/in -v /path/to/output/data:/out -v /path/to/model:/model  tofsegmentation:latest \
    --src_path /in --dst_path /out --model_path /model \
    --filter_series csBTFE_M2D --num_workers 4
```
* ***/path/to/input/data***: Local directory containing input MRI data.
* ***/path/to/output/data***: Local directory where the results will be saved.
* ***/path/to/model/file***: Local directory to the trained model file.

### Output Details:
The results will be organized in the output folder under the following structure:

Subfolder for each patient (e.g., patientID):

* ***Contour_images***: Contains JPG images of all slices with contours.
* ***predicted_TOF.nii.gz***: The 4D cine with predicted segmentation masks.

### Results file:
In the root of the output folder, you will find a CSV file named CMR_quantification_UZL.csv, containing quantification data with the following columns for each patient ID:

* patient
* time
* serie
* LV EDV [ml] (Left Ventricular End-Diastolic Volume)
* LV ESV [ml] (Left Ventricular End-Systolic Volume)
* LV EF [%] (Left Ventricular Ejection Fraction)
* LV mass [g] (Left Ventricular Mass)
* RV EDV [ml] (Right Ventricular End-Diastolic Volume)
* RV ESV [ml] (Right Ventricular End-Systolic Volume)
* RV EF [%] (Right Ventricular Ejection Fraction)
* RV mass [g] (Right Ventricular Mass)


Adjust the command and paths according to your specific data and model setup.








