#!/usr/bin/env python
"""
A simple script for converting a 4D NIfTI image into a set of RADAR files.
One RADAR file is generated for the affine matrix, and one RADAR file is generated for the volume arrays.
"""
import os
import sys
import argparse

import numpy as np
import nibabel as nib

import radar


def make_argparser() -> argparse.ArgumentParser:
    """
    Construct the `ArgumentParser` instance for the script.

    Returns
    -------
        An instance of `argparse.ArgumentParser` used to extract command-line arguments and display help and usage information.
    """
    this_script = os.path.basename(__file__)
    argp = argparse.ArgumentParser(prog=this_script, description='Convert 4D NIfTI images into RADAR format')
    argp.add_argument('-i', '--input' , type=str, action='store', help='The path and filename of the input NIfTI file (.nii or .nii.gz).'                                      , default=None, required=True)
    argp.add_argument('-o', '--output', type=str, action='store', help='The path to the output file. If the file exists, it is overwritten.'                                   , default=None, required=False)
    argp.add_argument('-O', '--order' , type=str, action='store', help='One of WHC, HWC, CWH or CHW specifying the meaning of the shape elements. If omitted, HWC is assumed.' , default=None, required=False)
    argp.add_argument('-f', '--format', type=str, action='store', help='One of F16, F32, F64 specifying the output data element format. If omitted, the source format is used.', default=None, required=False)
    return argp


def main() -> None:
    """
    The main entry point of the script.
    """
    argp = make_argparser()
    args, _fwd  = argp.parse_known_args()
    AFFINE_EXT  = '.affine' + radar.DEFAULT_EXTENSION
    VOLUME_EXT  = '.volume' + radar.DEFAULT_EXTENSION
    affine_path = None
    volume_path = None
    array_shape = None
    data_format = radar.DataFormat.UNKNOWN
    shape_order = radar.ShapeOrder.HWC_3D
    interpret   = radar.DataInterpretation.SCALAR
    storage     = radar.StorageLayout.CONTIGUOUS

    # Process the path to the input file.
    args.input = os.path.abspath(args.input)
    if not os.path.isfile(args.input):
        print(f'ERROR: The input NIfTI file could not be found at {args.input}.')
        argp.print_usage()
        sys.exit(1)

    # Generate the paths to the output files.
    if not args.output:
        dirname, filename = os.path.split(args.input)
        file   , _ext     = filename.split(os.extsep, 1)
        affine_path       = os.path.join(dirname, file + AFFINE_EXT)
        volume_path       = os.path.join(dirname, file + VOLUME_EXT)
    else:
        args.output       = os.path.abspath(args.output)
        dirname, filename = os.path.split(args.output)
        file   , ext      = filename.split(os.extsep, 1)
        affine_path       = os.path.join(dirname, file + '.affine.' + ext)
        volume_path       = os.path.join(dirname, file + '.volume.' + ext)

    # Extract the shape ordering.
    if args.order is not None:
        shape_order = radar.parse_shape_order(args.order, default=None, raise_error=False)
    else:
        shape_order = radar.ShapeOrder.HWC_3D

    if shape_order is None:
        print(f'ERROR: The value \'{args.order}\' specified for the --order argument is not valid.')
        print(f'  Valid values for --order are WHC, HWC, CWH, or CHW.')
        argp.print_usage()
        sys.exit(1)

    # Extract the data element format (which may be different than the source format).
    if args.format is not None:
        data_format = radar.parse_data_format(args.format, default=None, raise_error=False)
    else:
        data_format = None

    # Load the input file.
    print(f'Loading input NIfTI file from {args.input}, {os.stat(args.input).st_size} byte(s).')
    nifti   = nib.load(args.input)
    affine  = nifti.affine.astype(dtype=np.float32)
    volumes = nifti.get_fdata(caching='unchanged')
    if affine is None or volumes is None:
        print(f'ERROR: Failed to load the NIfTI input file from {args.input}.')
        sys.exit(1)
    if len(volumes.shape) < 4:
        print(f'ERROR: The input NIfTI file {args.input} must have 4 or more dimensions; found {len(volumes.shape)}.')
        sys.exit(1)

    # Extract the number of volumes and shape indices.
    w, h, c = radar.indices_for_shape_order(shape_order)
    n       = volumes.shape[3] # In NIfTI, time is always the 4th element.
    if data_format is None:
        data_format = radar.dtype_to_data_format(volumes.dtype, raise_error=False)
    if data_format == radar.DataFormat.UNKNOWN:
        print(f'ERROR: The input data format {volumes.dtype} is not supported by RADAR.')
        print(f'  Specify a value for the --format argument.')
        print(f'  Valid values for --format are {radar.DATA_FORMAT_NAMES}.')
        argp.print_usage()
        sys.exit(1)

    print(f'Found {n} volume(s) with dimensions (WxHxD) {volumes.shape[w]}, {volumes.shape[h]}, {volumes.shape[c]}.')

    # Generate the file headers.
    affine_v1, affine_header = radar.make_header_v1(radar.DataFormat.F32, radar.DataInterpretation.SCALAR, radar.StorageLayout.CONTIGUOUS, radar.ShapeOrder.WH_2D, array_shape=affine.shape, array_count=1)
    volume_v1, volume_header = radar.make_header_v1(data_format, interpret, storage, shape_order, array_shape=volumes.shape[0:3], array_count=n)

    # Ensure the output directory tree exists.
    output_dir, _ = os.path.split(affine_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write the affine output.
    with open(affine_path, mode='wb') as affine_fp:
        affine_offset  = affine_fp.write(affine_header)
        affine_offset += radar.append_array(affine_fp, affine, affine_v1, np.float32)

    # Write the volumes output.
    with open(volume_path, mode='wb') as volume_fp:
        volume_offset  = volume_fp.write(volume_header)
        for volume_index in range(n):
            volume_data: np.ndarray = volumes[:,:,:,volume_index]
            volume_offset += radar.append_array(volume_fp, volume_data, volume_v1)
        
    print(f'Wrote {os.stat(affine_path).st_size} byte(s) to file {affine_path}.')
    print(f'Wrote {os.stat(volume_path).st_size} byte(s) to file {volume_path}.')
    sys.exit(0)


if __name__ == "__main__":
    main()
