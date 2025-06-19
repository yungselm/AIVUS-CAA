import os
from os.path import join, split, isdir
from argparse import ArgumentParser

from deep_utils import DirUtils, NIBUtils


def dicom2nifti(nifti_dir: str, dicom_path: str):
    os.makedirs(nifti_dir, exist_ok=True)
    command = (f'dcm2niix -z y -b y -ba n -f %f_%t_%p_%s_%d_%i_%e_%q_%z_%m_%a_%g -o'
               f' "{nifti_dir}" "{dicom_path}"')
    output = os.system(command)
    if output != 0:
        raise ValueError(f"Input Dicom directory is not valid! nifti_dir: {nifti_dir}, dicom_dir: {dicom_path}")
    img_path = DirUtils.list_dir_full_path(nifti_dir, interest_extensions=".gz")[0]
    return img_path


parser = ArgumentParser()
parser.add_argument("--input_data", required=True, help="path the dicom directory or file")
parser.add_argument("--output_dir", default="output", help="Output directory where all the nifti files will be created")
parser.add_argument("--suffix", default="", help="A suffix to add to the output nifti files")
parser.add_argument("--is_dicom", action="store_true",
                    help="If set to true means that the output is a dicom and not a directory of dicom files")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.is_dicom:
        dicom2nifti(DirUtils.split_extension(args.input_data, suffix="_nifti"), args.input_data)
    else:
        if isdir(args.input_data):
            samples = DirUtils.list_dir_full_path(args.input_data, interest_extensions=".gz")
        else:
            samples = [args.input_data]
        for sample in samples:
            sample_name = split(sample)[-1].replace(".nii.gz", "")
            img_arr, img_img = NIBUtils.get_array_img(sample)

            for index in range(img_arr.shape[-1]):
                img_arr_index = img_arr[..., index: index + 1]
                output_img_path = join(args.output_dir, sample_name,
                                       f"{sample_name}_frame_{index}{args.suffix}_img.nii.gz")
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                NIBUtils.save_sample(output_img_path, img_arr_index, nib_img=img_img)
