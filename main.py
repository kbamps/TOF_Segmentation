import json
import multiprocessing
import os
from pathlib import Path
import sys
import cv2
import nibabel as nib
import glob
import pandas as pd
import pydicom
import numpy as np
import scipy.io
import time
import csv


from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftyFileModality
from dicomorganizer.utils import create_dicommanager_filter, extract_format
sys.path.append(os.getcwd())
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.factories.directory_structure import MircStructure
from tensorflow.keras.utils import to_categorical

from dicomorganizer import DicomManager

## might be required to read some dicomimages
import pylibjpeg
#import GDCM

def slice2bgr(slice_data):
    """
    Converts a 2D slice of grayscale data into a BGR color image.

    Parameters
    ----------
    slice_data : numpy.ndarray
        A 2D numpy array representing a single slice of grayscale data. The values in the array are assumed to be in 
        the range of the image intensity (e.g., [0, 255] or [0, 1]).

    Returns
    -------
    numpy.ndarray
        A 3D numpy array representing the input slice converted into a BGR color image. The dimensions of the output 
        image will be (height, width, 3), where 3 corresponds to the three color channels (blue, green, red).
    """
    
    # Normalize the data (optional, adjust as needed)
    normalized_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
    normalized_data = normalized_data.astype(np.uint8)

    # Create a bgr image from the slice data
    image = cv2.cvtColor(normalized_data, cv2.COLOR_GRAY2BGR)

    return image


def convert2png(original_image_paths, predicted_image_paths, out_path):
    """
    Converts 3D medical images into 2D PNG slices, draws contours of predicted class regions on the original images, 
    and saves the resulting images to the specified output directory.

    Parameters
    ----------
    original_image_paths : list of str
        A list of file paths to the original 3D medical images in NIfTI format. These images will be used as the base 
        for contour drawing.
    predicted_image_paths : list of str
        A list of file paths to the predicted 3D images in NIfTI format. The predicted images should contain class labels 
        for segmentation, where each pixel value corresponds to a class label.
    out_path : str
        The directory where the resulting PNG images will be saved. Contoured slices will be saved in a subdirectory 
        called 'contour_images' under this path.

    Returns
    -------
    None
        The function does not return anything. It saves the generated PNG images to the specified output directory.
    """
    
    # Define the colors for each class
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR format
    instance = 0
    for org_path, pred_path in zip(original_image_paths, predicted_image_paths):
        org_data = nib.load(org_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()
        

        for z in np.arange(org_data.shape[2]):
            org_slice_data = slice2bgr(org_data[:,:,z])
            pred_slice_data = pred_data[:,:,z]


            for class_label in range(1, int(pred_slice_data.max())):
                # Create a binary mask for the current class
                class_mask = (pred_slice_data == class_label).astype(np.uint8)

                # Find contours for the binary mask
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw the contours on the contour_image with the corresponding color
                cv2.drawContours(org_slice_data, contours, -1, colors[class_label-1], thickness=1)


            # Define the output PNG file name
            name = str(instance).zfill(4)
            out_png_images = os.path.join(out_path, 'contour_images')
            os.makedirs(out_png_images, exist_ok=True)
            output_filename = os.path.join(out_png_images, f'{name}.png')

            # Save the image as a PNG file using OpenCV
            cv2.imwrite(output_filename, org_slice_data)
            instance += 1


def predict_on_test(test_data, model_base_directory):
    """
    Performs ensemble prediction on test data using trained models from multiple folds, 
    averages the predictions, and saves the aggregated results as NIfTI files.

    Parameters
    ----------
    test_data : object
        The test dataset to be used for prediction. This should be compatible with the `MircSampler` class.
    model_base_directory : str
        The base directory where the model files for different folds are stored.

    Returns
    -------
    list of str
        A list of file paths to the saved NIfTI files containing the averaged predictions 
        across all folds for each subject in the test data.
    """
    test_sampler = MircSampler(test_data)
    predictions = []
    for fold_i in range(5):
        output_structure = MircStructure(
            base_dir=os.path.join(model_base_directory, 'models'),
            run_name='final',
            experiment_name='ACDC_TOF',
            fold_i=fold_i,
            round_i=0,  # when None a new round will be created
            testing_mirc=test_data
        )
        dvn_model = DvnModel.load_model(os.path.join(output_structure.models_dir, "dvn_model_final")) # Sommige folds falen met tf2, maar alles werkt met tf2.7!!
        print("model fold {} round {} loaded".format(fold_i,0))
        predictions.append(dvn_model.predict("full_test", test_sampler))

    prediction_names = []
    for subject in range(len(predictions[0])):
        prediction = predictions[0][subject][0][0][0,:,:,:,:].astype('float32')
        for fold in range(1,5):
            prediction += predictions[fold][subject][0][0][0,:,:,:,:]
        prediction = prediction / 5

        filename = output_structure.test_images_output_dirs[subject].replace('Fold_4','').replace('Round_0_','') + "allfolds.nii.gz"
        head,tail = os.path.split(filename)
        if not os.path.exists(head):
            os.makedirs(head)
        nib.save(nib.Nifti1Image(prediction, predictions[0][subject][0][0].affine[0]), filename)

        prediction_names.append(filename)


    return(prediction_names)



def predict_patient(patient_id, ref_spacing, ref_size, csv_file, dcminfo, out_path, model_path):
    """
    Processes DICOM images for a specific patient, resamples them to a reference spacing and size, 
    applies a trained model for cardiac segmentation, and computes cardiac metrics such as volumes 
    and ejection fractions. Results are saved in multiple formats including NIfTI, CSV, and JSON.

    Parameters
    ----------
    patient_id : str
        The identifier for the patient being processed.
    ref_spacing : list or numpy.ndarray
        The target voxel spacing for resampling, provided as [x_spacing, y_spacing, z_spacing].
    ref_size : list or numpy.ndarray
        The desired output dimensions [x_size, y_size, z_size] of the resampled volume.
    csv_file : str
        Path to the CSV file where calculated cardiac metrics (e.g., EDV, EF) will be appended.
    dcminfo : pandas.DataFrame
        DataFrame containing metadata about the patient's DICOM files, including `filename`, 
        `InstanceNumber`, and `SliceLocation`.
    out_path : str
        Directory where preprocessed images, predictions, and results will be saved.
    model_path : str
        Path to the trained model used for cardiac segmentation.

    Returns
    -------
    None
        Saves the results directly to the specified output path and appends metrics to the CSV file.
    """

    start_time_patient = time.time()
    print(f"Processing patient {patient_id}...")
    print(f"Number of DICOM files: {len(dcminfo)}")


    case_ = patient_id
    date_ = dcminfo.iloc[0].AcquisitionDate
    dcminfo = dcminfo.sort_values(by='InstanceNumber')
    I = np.transpose(np.array([pydicom.dcmread(x).pixel_array for x in dcminfo["filename"]]),[1,2,0])
    slicelocations = dcminfo["SliceLocation"].unique()
    no_slices = len(slicelocations)
    no_times = int(len(dcminfo)/no_slices)
    I = np.reshape(I,[I.shape[0],I.shape[1],no_slices,no_times])
    info_slices = dcminfo[::no_times]
    slicelocations = info_slices["SliceLocation"].values
    I = I[:,:,np.argsort(slicelocations),:]
    I = I[:, :, ::-1, :]



    try:
        spacing_between_slices = dcminfo.iloc[0].SpacingBetweenSlices if dcminfo.iloc[0].SpacingBetweenSlices != "" else np.abs(slicelocations[1]-slicelocations[0])
    except:
        spacing_between_slices = np.abs(slicelocations[1]-slicelocations[0])
    vox_spacing = np.array([dcminfo.iloc[0].PixelSpacing[0],dcminfo.iloc[0].PixelSpacing[1],spacing_between_slices])

    # resample images
    deviating_spacing = np.abs(1 - vox_spacing/ref_spacing) > 0.05
    ref_spacing_patient = np.array(vox_spacing)
    ref_spacing_patient[deviating_spacing] = ref_spacing[deviating_spacing]
    shape_restore = np.array([I.shape[0],I.shape[1],I.shape[2]])
    rot_origin = (shape_restore+1) / 2
    S = np.array([[ref_spacing_patient[0]/vox_spacing[1],0,0,0],
                    [0,ref_spacing_patient[1] / vox_spacing[0],0,0],
                    [0,0,ref_spacing_patient[2] / vox_spacing[2],0],
                    [0, 0, 0, 1]])
    T = np.array([[1,0,0,rot_origin[0]],
                        [0,1,0,rot_origin[1]],
                        [0,0,1,rot_origin[2]],
                        [0,0,0,1]])
    T_ = np.array([[1,0,0,-(ref_size[0]+1)/2],
                        [0,1,0,-(ref_size[1]+1)/2],
                        [0,0,1,-(ref_size[2]+1)/2],
                        [0,0,0,1]])
    A = np.matmul(T,np.matmul(S,T_))

    affine = np.array([[ref_spacing_patient[0], 0, 0, ref_spacing_patient[0]],
                    [0, ref_spacing_patient[1], 0, ref_spacing_patient[1]],
                    [0, 0, ref_spacing_patient[2], ref_spacing_patient[2]],
                    [0, 0, 0, 1]])

    affine_orig = [[vox_spacing[0], 0, 0, vox_spacing[0]],
                    [0, vox_spacing[1], 0, vox_spacing[1]],
                    [0, 0, vox_spacing[2], vox_spacing[2]],
                    [0, 0, 0, 1]]
    
    [Y,X,Z] = np.meshgrid(np.linspace(1,  ref_size[0], ref_size[0]),np.linspace(1,  ref_size[1], ref_size[1]),np.linspace(1,  ref_size[2], ref_size[2]))
    XT = np.array([np.ndarray.flatten(X),np.ndarray.flatten(Y),np.ndarray.flatten(Z),np.ones((np.prod(ref_size)))])

    Xnew = np.matmul(A, XT)
    x_new = np.around(np.reshape(Xnew[0], (ref_size[0], ref_size[1], ref_size[2])), decimals=3)
    y_new = np.around(np.reshape(Xnew[1], (ref_size[0], ref_size[1], ref_size[2])), decimals=3)
    z_new = np.around(np.reshape(Xnew[2], (ref_size[0], ref_size[1], ref_size[2])), decimals=3)

        # custom_ip
    z_slices = z_new[0,0,:]
    z_slices_ref = np.unique(np.round(z_slices))
    pos = np.argmin(np.fliplr(np.abs(np.reshape(z_slices,[1,z_slices.shape[0]]) - np.reshape(z_slices_ref,[z_slices_ref.shape[0],1]))),axis=1)
    pos = 47 - pos
    z_slices[pos] = z_slices_ref
    z_new= np.tile(z_slices, [ref_size[0],ref_size[1],1])

        # create data structure for predictions
    preprocessed_image_dir = os.path.join(out_path,'preprocessed_images')
    if not os.path.exists(preprocessed_image_dir):
        os.makedirs(preprocessed_image_dir)
    dataset_name = "TOF_predict_new_cases"
    dataset = Dataset(dataset_name,preprocessed_image_dir)


    for time_ in range(I.shape[3]):
        
        I_resampled = scipy.ndimage.map_coordinates(np.float32(I[:, :, :, time_]), (x_new-1, y_new-1, z_new-1), order=1, mode='constant', cval=np.nan)

        mask = ~np.isnan(I_resampled)
        tmp = np.ndarray.flatten(I_resampled)
        tmp = tmp[~np.isnan(tmp)]
        Q1 = np.quantile(tmp,0.01)
        Q99 = np.quantile(tmp,0.99)
        I_resampled[I_resampled<Q1]=Q1
        I_resampled[I_resampled > Q99] = Q99
        I_resampled[~mask]=0

        I_resampled = (I_resampled-np.mean(I_resampled))/np.std(I_resampled)

        case_name = '{}_serie_{:04d}_time_{:02d}'.format(case_,dcminfo.iloc[0].SeriesNumber,time_+1)
        image_name = os.path.join(preprocessed_image_dir,dataset_name,case_name + '.nii.gz')
        head, tail = os.path.split(image_name)
        if not os.path.exists(head):
            os.makedirs(head)

        nib.save(nib.Nifti1Image(I_resampled, affine), image_name)
        nib.save(nib.Nifti1Image(np.float32(I[:, :, :, time_]), affine_orig), os.path.join(preprocessed_image_dir,dataset_name, f"original_{str(time_+1).zfill(2)}" + '.nii.gz'))

        case = Case(case_name)
        record = Record("record_0")
        record.add(NiftyFileModality("MR", image_name))
        case.add(record)
        dataset.add(case)

        test_data = Mirc()
        test_data.add(dataset)

    prediction_files = predict_on_test(test_data, model_path)

    AT = np.linalg.inv(A)
    [YR, XR, ZR] = np.meshgrid(np.linspace(1, shape_restore[0], shape_restore[0]),
                                np.linspace(1, shape_restore[1], shape_restore[1]),
                                np.linspace(1, shape_restore[2], shape_restore[2]))
    XRT = np.array([np.ndarray.flatten(XR), np.ndarray.flatten(YR), np.ndarray.flatten(ZR), np.ones((np.prod(shape_restore)))])

    XRT_new = np.matmul(AT, XRT)
    x_new = np.around(np.reshape(XRT_new[0], (shape_restore[0], shape_restore[1], shape_restore[2])), decimals=3)
    y_new = np.around(np.reshape(XRT_new[1], (shape_restore[0], shape_restore[1], shape_restore[2])), decimals=3)
    z_new = np.round(np.reshape(XRT_new[2], (shape_restore[0], shape_restore[1], shape_restore[2])))

    pred_orig_cat = []
    for nf,file in enumerate(prediction_files):
        img = nib.load(file)
        pred = img.get_fdata()

            # resample back to original spacing

        pred_orig = []
        for k in range(5):
            pred_orig.append(scipy.ndimage.map_coordinates(np.float32(pred[:,:,:,k]), (x_new-1, y_new-1, z_new-1), order=1, mode='constant', cval=0))

        pred_orig = np.argmax(np.array(pred_orig), axis=0)
        filename = file.replace('models/run_1', 'results').replace('/record_0/Testing/ACDC_TOF_allfolds', '')
        filename = os.path.basename(filename)
        filename = os.path.join(out_path, 'predicted_images', f"{nf}_{filename}")
        head, tail = os.path.split(filename)
        if not os.path.exists(head):
            os.makedirs(head)
        nib.save(nib.Nifti1Image(np.float32(pred_orig), affine_orig), filename)

        # calculate volumes
        pred_orig_cat.append(to_categorical(pred_orig, num_classes=5))

    pred_orig_cat = np.array(pred_orig_cat)
    volumes = np.sum(pred_orig_cat[:,:,:,:,1:],axis=(1,2,3) ) * np.prod(vox_spacing) / 10e2

    nib.save(nib.Nifti1Image(volumes, affine_orig), os.path.join(out_path, 'contours_AI.nii.gz'))

    LV_volumes = volumes[:,0]
    RV_volumes = volumes[:,2]
    LV_masses = volumes[:,1] * 1.05
    RV_masses = volumes[:,3] * 1.05

    LV_phases = [np.argmax(LV_volumes),np.argmin(LV_volumes)]
    RV_phases = [np.argmax(RV_volumes), np.argmin(RV_volumes)]

    EDV_LV = LV_volumes[LV_phases[0]]
    ESV_LV = LV_volumes[LV_phases[1]]
    EDV_RV = RV_volumes[RV_phases[0]]
    ESV_RV = RV_volumes[RV_phases[1]]

    EF_LV = (EDV_LV - ESV_LV) / EDV_LV *100
    EF_RV = (EDV_RV - ESV_RV) / EDV_RV *100

    MASS_LV = LV_masses[LV_phases[0]]
    MASS_RV = RV_masses[RV_phases[0]]

    print('patient_{}_serie_{:4d}_predicted in {:.2f} seconds'.format(case_,dcminfo.iloc[0].SeriesNumber,time.time()-start_time_patient))
    print('LV: EDV {:.2f}ml ESV: {:.2f}ml EF: {:.2f}% MASS: {:.2f}g'.format(EDV_LV, ESV_LV, EF_LV, MASS_LV))
    print('RV: EDV {:.2f}ml ESV: {:.2f}ml EF: {:.2f}% MASS: {:.2f}g'.format(EDV_RV, ESV_RV, EF_RV, MASS_RV))

    fields = [case_,date_,dcminfo.iloc[0].SeriesDescription, EDV_LV, ESV_LV, EF_LV, MASS_LV, EDV_RV, ESV_RV, EF_RV, MASS_RV]
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)


    # Create a dictionary for the data
    data = {
        'patient': case_,
        'time': date_,
        'serie': dcminfo.iloc[0].SeriesDescription,
        'LV_EDV': EDV_LV.item(),
        'LV_ESV': ESV_LV.item(),
        'LV_EF': EF_LV.item(),
        'LV_mass': MASS_LV.item(),
        'RV_EDV': EDV_RV.item(),
        'RV_ESV': ESV_RV.item(),
        'RV_EF': EF_RV.item(),
        'RV_mass': MASS_RV.item()
    }

    # # Save the list of data dictionaries as a JSON file
    # json_file = os.path.join(args.dst_path,'results','meta_results.json')
    # with open(json_file, 'w') as file:
    #     json.dump(data, file, indent=4)

    ## postprocessing convert contours to png
    # original_image_paths = glob.glob(os.path.join(preprocessed_image_dir,"*", "original_*"))
    # predicted_image_paths = glob.glob(os.path.join(out_path,'predicted_images', "*.nii.gz"))
    # convert2png(original_image_paths, predicted_image_paths,out_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    import argparse
    import traceback
    import json

    parser = argparse.ArgumentParser(description="Tetralogy of Fallot - TOF: Segmentation and quantification of cardiac MRI data of TOF.")
    parser.add_argument('--src_path', type=str, required=True, help="Path to the source directory containing the MRI input data.")
    parser.add_argument('--dst_path', type=str, required=True, help="Path to the destination directory where results will be saved.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file used for processing.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use for processing. Defaults to 1.")
    parser.add_argument('--gpu', type=int, default=0, 
                        help="GPU ID to use for processing. Set to -1 to use the CPU.")
    parser.add_argument(
        "--filters", 
        nargs='*', 
        help="Filters in the format key1=value1 key2=value2 ...",
        default=[]
    )

    args = parser.parse_args()

    # Set the GPU or CPU for processing
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Using CPU for processing.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu} for processing.")

    
    ref_spacing = np.array([1.5,1.5,3])
    ref_size = np.array([192,192,48])

    patient_dir = os.path.join(args.src_path,os.listdir(args.src_path)[0])

    csv_file = os.path.join(args.dst_path,'CMR_quantification_UZL.csv')
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    if not os.path.exists(csv_file):
        fields = ['patient','time','serie','LV EDV [ml]','LV ESV [ml]','LV EF [%]','LV mass [g]','RV EDV [ml]','RV ESV [ml]','RV EF [%]','RV mass [g]']
        with open(csv_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(fields)

    manager = DicomManager(directory=args.src_path, tags= ["ProtocolName","StudyDescription", "StudyID", "PatientName","SeriesInstanceUID", "StudyDate","PatientID", "SeriesDescription","SeriesNumber", "AcquisitionTime", "SliceLocation", "AcquisitionDate", "PixelSpacing", "InstanceNumber", "Modality"], num_workers=args.num_workers, group_by='SeriesInstanceUID')

    filters = args.filters
    filters.extend(["AcquisitionTime=.*", "SliceLocation=.*"])
    filter_by = create_dicommanager_filter(filters)
    instances_before_filtering = len(manager.df_dicom.obj)
    manager.filter(filter_by)
    instances_after_filtering = len(manager.df_dicom.obj)
    filters_str=",".join(filters)
    print(f"Applying filters: {filters_str}")
    print(f"Number of instances before filtering: {instances_before_filtering}")
    print(f"Number of instances after filtering: {instances_after_filtering}")



    output_format_path = Path(args.dst_path) / '$PatientID$/$Modality$/$StudyDate$/$SeriesNumber$_$SeriesDescription$'

    if not (output_format_path / "DCM").exists:
        result = manager.export_to_folder_structure(output_format_path / "DCM")

        if len(result["failed"]) > 0:
            print("----------------------------------------------------------------")
            print("The following DICOM files failed to be processed and copied:")
            for failed_file in result["failed"]:
                print(failed_file)
            print("----------------------------------------------------------------")
    else:
        print("----------------------------------------------------------------")
        print("The DICOM files have already been processed and copied to the destination directory.")
        print("----------------------------------------------------------------")


    if len(manager.df_dicom.obj) == 0:
        print("----------------------------------------------------------------")
        print("No DICOM files found in the specified directory.")
        print("Check if the description in 'filter_series' is correct.")
        print("Check if the directory contains the DICOM files.")
        print("----------------------------------------------------------------")
        print()
        print("Exiting the program...")
        sys.exit(0)

    # Print an overview of the patients found in the dataframe
    print("----------------------------------------------------------------")
    print("Overview of series found:")
    for series_id, df_dicom_series in manager.df_dicom:
        print(f"Series ID: {series_id}")
    print("----------------------------------------------------------------")

    for series_id, df_dicom_series in manager.df_dicom:
        out_path = extract_format(output_format_path.as_posix(), df_dicom_series.iloc[0].to_dict())
        patient_id = df_dicom_series.iloc[0].PatientID
        os.makedirs(out_path, exist_ok=True)
        try:
            predict_patient(patient_id, ref_spacing, ref_size, csv_file, df_dicom_series, out_path, args.model_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Prediction for patient {patient_id} failed: {e}")
            fields = [patient_id]
            with open(csv_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(fields)
        print("----------------------------------------------------------------")

    print('All patients predicted')


    # print all patients from dicommanager to json
    # Dump the required fields for each series in the DICOM manager as a JSON to the command prompt
    dicom_data = []
    for series_id, df_dicom_series in manager.df_dicom:
        first_row = df_dicom_series.iloc[0]
        dicom_data.append({
            "patient_name": str(first_row.get("PatientName", "N/A")),
            "patient_id": str(first_row.get("PatientID", "N/A")),
            "study_id": str(first_row.get("StudyID", "N/A")),
            "study_description": first_row.get("StudyDescription", "N/A"),
            "study_date": str(first_row.get("StudyDate", "N/A")),
            "acquisition_date": str(first_row.get("AcquisitionDate", "N/A")),
            "protocol": first_row.get("ProtocolName", "N/A"),
            "modality": first_row.get("Modality", "N/A"),
            "series_number": int(first_row.get("SeriesNumber", -1)) if not pd.isna(first_row.get("SeriesNumber")) else "N/A",
            "series_description": first_row.get("SeriesDescription", "N/A"),
            "series_instance_uid": str(first_row.get("SeriesInstanceUID", "N/A"))
        })
    print("###---###")
    # Print the JSON to the command prompt
    print(json.dumps(dicom_data))
    print("###---###")







