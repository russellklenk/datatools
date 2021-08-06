#!/usr/bin/env python
"""
Simple utility to dump the header fields of a RADAR file.
"""
import os
import sys
import argparse

import radar


def make_argparser() -> argparse.ArgumentParser:
    """
    Construct the `ArgumentParser` instance for the script.

    Returns
    -------
        An instance of `argparse.ArgumentParser` used to extract command-line arguments and display help and usage information.
    """
    this_script = os.path.basename(__file__)
    argp = argparse.ArgumentParser(prog=this_script, description='Dump the header fields of a RADAR file')
    argp.add_argument('-i', '--input' , type=str, action='store', help='The path and filename of the input RADAR file (.radar).', default=None, required=True)
    return argp


def main() -> None:
    """
    The main entry point of the script.
    """
    argp = make_argparser()
    args, _fwd = argp.parse_known_args()

    args.input = os.path.abspath(args.input)
    if not os.path.isfile(args.input):
        print(f'ERROR: The input RADAR file could not be found at {args.input}.')
        argp.print_usage()
        sys.exit(1)

    with open(args.input, mode='rb') as radar_fp:
        header, _header_bytes = radar.load_header_v1(radar_fp)
        if not header:
            print(f'ERROR: Input file {args.input} does not have a valid RADAR header.')
            sys.exit(1)

        print(f"MAGIC         : {header.magic.decode('ascii')}")
        print(f'VERSION       : {header.version} ')
        print(f'STORAGE       : {radar.storage_layout_to_str(header.storage_layout)}')
        print(f'SHAPE ORDER   : {radar.shape_order_to_str(header.shape_order)}')
        print(f'DATA FORMAT   : {radar.data_format_to_str(header.data_format)}')
        print(f'INTERPRETATION: {radar.data_interpretation_to_str(header.interpretation)}')
        print(f'ELEMENT STRIDE: {header.element_stride} byte(s)')
        print(f'X STRIDE      : {header.x_stride} byte(s)')
        print(f'ARRAY STRIDE  : {header.array_stride} byte(s)')
        print(f'ARRAY COUNT   : {header.array_count}')
        print(f'ARRAY SHAPE   : {header.array_shape}')
        print('')

    sys.exit(0)


if __name__ == "__main__":
    main()
