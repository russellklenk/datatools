"""
This module provides access to types and functions for reading and writing Raw Array (RADR) data files.
The standard file extension is `.radar`, which stands for Raw Array DAta Record.
"""
import os
import io
import struct

from   enum   import IntEnum
from   typing import ByteString, Dict, List, Optional, Tuple, Union

import numpy as np


MAGIC_BYTES      : bytes = 'RADR'.encode('ascii') # The magic bytes appearing at the start of any Raw Array binary data file.
DEFAULT_EXTENSION: str   = '.radar'               # The default file extension for files in RADR format.
MAXIMUM_VERSION  : int   =  1                     # The maximum supported version of the RADR binary data format.
MAXIMUM_DIMENSION: int   =  65536                 # The maximum size of any one dimension of array data.
HEADER_SIZE_V1   : int   =  32                    # The number of bytes allocated for the Raw Array binary data header in version 1 of the format.


class DataFormat(IntEnum):
    """
    Define the data element types supported by the Raw Array format.
    All data element values are stored using Little Endian byte ordering, with the least-significant bit first.

    Fields
    ------
        UNKNOWN: The data element type is not known, not supported, or cannot be determined.
        S8     : Data elements are signed 8-bit integers.
        U8     : Data elements are unsigned 8-bit integers.
        S16    : Data elements are signed 16-bit integers.
        U16    : Data elements are unsigned 16-bit integers.
        S32    : Data elements are signed 32-bit integers.
        U32    : Data elements are unsigned 32-bit integers.
        S64    : Data elements are signed 64-bit integers.
        U64    : Data elements are unsigned 64-bit integers.
        F16    : Data elements are IEEE 754.2008 binary16 half-precision floating point values.
        F32    : Data elements are IEEE 754 single-precision floating point values.
        F64    : Data elements are IEEE 754 double-prevision floating point values.
    """
    UNKNOWN              =  0
    S8                   =  1
    U8                   =  2
    S16                  =  3
    U16                  =  4
    S32                  =  5
    U32                  =  6
    S64                  =  8
    U64                  =  9
    F16                  = 10
    F32                  = 11
    F64                  = 12


DATA_FORMAT_NAMES   : List[str] = [
    's8' , 'int8'   , 'sint8' , 'char' ,
    'u8' , 'uint8'  , 'byte'  ,
    's16', 'int16'  , 'sint16', 'short',
    'u16', 'uint16' , 'ushort',
    's32', 'int32'  , 'sint32', 'int'  ,
    'u32', 'uint32' , 'uint'  ,
    's64', 'int64'  , 'sint64', 'long' ,
    'u64', 'uint64' , 'ulong' ,
    'f16', 'float16', 'half'  ,
    'f32', 'float32', 'float' , 'single',
    'f64', 'float64', 'double'
]

DATA_FORMAT_LOOKUP  : Dict[str, DataFormat] = {
    DATA_FORMAT_NAMES[ 0]: DataFormat.S8,
    DATA_FORMAT_NAMES[ 1]: DataFormat.S8,
    DATA_FORMAT_NAMES[ 2]: DataFormat.S8,
    DATA_FORMAT_NAMES[ 3]: DataFormat.S8,

    DATA_FORMAT_NAMES[ 4]: DataFormat.U8,
    DATA_FORMAT_NAMES[ 5]: DataFormat.U8,
    DATA_FORMAT_NAMES[ 6]: DataFormat.U8,

    DATA_FORMAT_NAMES[ 7]: DataFormat.S16,
    DATA_FORMAT_NAMES[ 8]: DataFormat.S16,
    DATA_FORMAT_NAMES[ 9]: DataFormat.S16,
    DATA_FORMAT_NAMES[10]: DataFormat.S16,

    DATA_FORMAT_NAMES[11]: DataFormat.U16,
    DATA_FORMAT_NAMES[12]: DataFormat.U16,
    DATA_FORMAT_NAMES[13]: DataFormat.U16,

    DATA_FORMAT_NAMES[14]: DataFormat.S32,
    DATA_FORMAT_NAMES[15]: DataFormat.S32,
    DATA_FORMAT_NAMES[16]: DataFormat.S32,
    DATA_FORMAT_NAMES[17]: DataFormat.S32,

    DATA_FORMAT_NAMES[18]: DataFormat.U32,
    DATA_FORMAT_NAMES[19]: DataFormat.U32,
    DATA_FORMAT_NAMES[20]: DataFormat.U32,

    DATA_FORMAT_NAMES[21]: DataFormat.S64,
    DATA_FORMAT_NAMES[22]: DataFormat.S64,
    DATA_FORMAT_NAMES[23]: DataFormat.S64,
    DATA_FORMAT_NAMES[24]: DataFormat.S64,

    DATA_FORMAT_NAMES[25]: DataFormat.U64,
    DATA_FORMAT_NAMES[26]: DataFormat.U64,
    DATA_FORMAT_NAMES[27]: DataFormat.U64,

    DATA_FORMAT_NAMES[28]: DataFormat.F16,
    DATA_FORMAT_NAMES[29]: DataFormat.F16,
    DATA_FORMAT_NAMES[30]: DataFormat.F16,

    DATA_FORMAT_NAMES[31]: DataFormat.F32,
    DATA_FORMAT_NAMES[32]: DataFormat.F32,
    DATA_FORMAT_NAMES[33]: DataFormat.F32,
    DATA_FORMAT_NAMES[34]: DataFormat.F32,

    DATA_FORMAT_NAMES[35]: DataFormat.F64,
    DATA_FORMAT_NAMES[36]: DataFormat.F64,
    DATA_FORMAT_NAMES[37]: DataFormat.F64
}

def parse_data_format(value: str, default: DataFormat=DataFormat.UNKNOWN, raise_error: bool=False) -> DataFormat:
    """
    Attempt to parse a string value into a member of the `DataFormat` enumeration.

    Parameters
    ----------
        value      : The `str` value to parse.
        default    : The value to return if `value` is not one of the values in `radar.DATA_FORMAT_NAMES`.
        raise_error: Specify `True` to raise a `ValueError` if `value` is not one of the values in `radar.DATA_FORMAT_NAMES`.

    Returns
    -------
        One of the values of the `DataFormat` enumeration, or `default` if `value` is not known.
    """
    if not value:
        if raise_error:
            raise ValueError('The value argument is None, or is an empty string')
        else:
            return default

    enum_value: Optional[DataFormat] = DATA_FORMAT_LOOKUP.get(value.lower(), None)
    if enum_value is not None:
        return enum_value
    elif raise_error:
        raise ValueError(f'The value argument \'{value}\' is not one of the recognized values {DATA_FORMAT_NAMES}')
    else:
        return default


def data_format_to_str(format: Union[int, DataFormat]) -> str:
    """
    Convert a value of the `DataFormat` enumeration to a descriptive string.

    Parameters
    ----------
        format: One of the values of the `DataFormat` enumeration.

    Returns
    -------
        A `str` description of the data format.
    """
    if format == DataFormat.UNKNOWN:
        return 'UNKNOWN'
    elif format == DataFormat.S8:
        return 'S8'
    elif format == DataFormat.U8:
        return 'U8'
    elif format == DataFormat.S16:
        return 'S16'
    elif format == DataFormat.U16:
        return 'U16'
    elif format == DataFormat.S32:
        return 'S32'
    elif format == DataFormat.U32:
        return 'U32'
    elif format == DataFormat.S64:
        return 'S64'
    elif format == DataFormat.U64:
        return 'U64'
    elif format == DataFormat.F16:
        return 'F16'
    elif format == DataFormat.F32:
        return 'F32'
    elif format == DataFormat.F64:
        return 'F64'
    else:
        return f'UNKNOWN ({int(format)})'
    

class DataInterpretation(IntEnum):
    """
    Define some well-known values used to specify how data in an array is interpreted.

    Fields
    ------
        UNSPECIFIED: The data layout is unspecified and should be interpreted as raw values of the given `DataFormat`. For example, to specify an opaque byte array, use `DataFormat.U8` and `DataInterpretation.UNSPECIFIED`.
        SCALAR     : The data should be interpreted as a set of scalar values, with each data element corresponding to a single value of the given type.
        VEC2       : The data should be interpreted as a 2-component vector, with one data element specified for each component (that is, each logical value is comprised of 2 data elements).
        VEC3       : The data should be interpreted as a 3-component vector, with one data element specified for each component (that is, each logical value is comprised of 3 data elements).
        VEC4       : The data should be interpreted as a 4-component vector, with one data element specified for each component (that is, each logical value is comprised of 4 data elements).
        RGB        : The data should be interpreted as an RGB color value, with one data element specified for each component (that is, each logical value is comprised of 3 data elements).
        RGBX       : The data should be interpreted as an RGB color value, with one data element specified for each component (that is, each logical value is comprised of 4 data elements, with the last data element being ignored).
        RGBA       : The data should be interpreted as an RGBA color value, with one data element specified for each component (that is, each logical value is comprised of 4 data elements).
    """
    UNSPECIFIED =  0
    SCALAR      =  1
    VEC2        =  2
    VEC3        =  3
    VEC4        =  4
    RGB         =  5
    RGBX        =  6
    RGBA        =  7


DATA_INTERPRETATION_NAMES : List[str] = ['unspecified', 'raw' , 'any', 'store', 'scalar', 'vec2', 'vec3', 'vec4', 'rgb', 'rgbx', 'rgba']
DATA_INTERPRETATION_LOOKUP: Dict[str, DataInterpretation] = {
    DATA_INTERPRETATION_NAMES[ 0]: DataInterpretation.UNSPECIFIED,
    DATA_INTERPRETATION_NAMES[ 1]: DataInterpretation.UNSPECIFIED,
    DATA_INTERPRETATION_NAMES[ 2]: DataInterpretation.UNSPECIFIED,
    DATA_INTERPRETATION_NAMES[ 3]: DataInterpretation.UNSPECIFIED,
    DATA_INTERPRETATION_NAMES[ 4]: DataInterpretation.SCALAR,
    DATA_INTERPRETATION_NAMES[ 5]: DataInterpretation.VEC2,
    DATA_INTERPRETATION_NAMES[ 6]: DataInterpretation.VEC3,
    DATA_INTERPRETATION_NAMES[ 7]: DataInterpretation.VEC4,
    DATA_INTERPRETATION_NAMES[ 8]: DataInterpretation.RGB,
    DATA_INTERPRETATION_NAMES[ 9]: DataInterpretation.RGBX,
    DATA_INTERPRETATION_NAMES[10]: DataInterpretation.RGBA
}


def parse_data_interpretation(value: str, default: DataFormat=DataFormat.UNKNOWN, raise_error: bool=False) -> DataInterpretation:
    """
    Attempt to parse a string value into a member of the `DataInterpretation` enumeration.

    Parameters
    ----------
        value      : The `str` value to parse.
        default    : The value to return if `value` is not one of the values in `radar.DATA_INTERPRETATION_NAMES`.
        raise_error: Specify `True` to raise a `ValueError` if `value` is not one of the values in `radar.DATA_INTERPRETATION_NAMES`.

    Returns
    -------
        One of the values of the `DataInterpretation` enumeration, or `default` if `value` is not known.
    """
    if not value:
        if raise_error:
            raise ValueError('The value argument is None, or is an empty string')
        else:
            return default

    enum_value: Optional[DataInterpretation] = DATA_INTERPRETATION_LOOKUP.get(value.lower(), None)
    if enum_value is not None:
        return enum_value
    elif raise_error:
        raise ValueError(f'The value argument \'{value}\' is not one of the recognized values {DATA_INTERPRETATION_NAMES}')
    else:
        return default


def data_interpretation_to_str(interpret: Union[int, DataInterpretation]) -> str:
    """
    Convert a value of the `DataInterpretation` enumeration to a descriptive string.

    Parameters
    ----------
        interpret: One of the values of the `DataInterpretation` enumeration.

    Returns
    -------
        A `str` description of the data interpretation.
    """
    if interpret == DataInterpretation.UNSPECIFIED:
        return 'UNSPECIFIED'
    elif interpret == DataInterpretation.SCALAR:
        return 'SCALAR'
    elif interpret == DataInterpretation.VEC2:
        return 'VEC2'
    elif interpret == DataInterpretation.VEC3:
        return 'VEC3'
    elif interpret == DataInterpretation.VEC4:
        return 'VEC4'
    elif interpret == DataInterpretation.RGB:
        return 'RGB'
    elif interpret == DataInterpretation.RGBX:
        return 'RGBX'
    elif interpret == DataInterpretation.RGBA:
        return 'RGBA'
    else:
        return f'UNKNOWN {int(interpret)}'


class ShapeOrder(IntEnum):
    """
    Define the supported layouts for 1D, 2D and 3D array data.

    Fields
    ------
        UNSPECIFIED: The shape has a single element with no specific interpretation.
        W_1D       : The shape has a single element and the data represents a row vector.
        H_1D       : The shape has a single element and the data represents a column vector.
        WH_2D      : The shape has two elements, with the first element specifying the number of data element columns, and the second element specifying the number of data element rows.
        HW_2D      : The shape has two elements, with the first element specifying the number of data element rows, and the second element specifying the number of data element columns.
        WHC_3D     : The shape has three elements, with the first element specifying the number of data element columns, the second element specifying the number of data element rows, and the third element specifying the number of channels or slices.
        HWC_3D     : The shape has three elements, with the first element specifying the number of data element rows, the second element specifying the number of data element columns, and the third element specifying the number of channels or slices.
        CWH_3D     : The shape has three elements, with the first element specifying the number of channels or slices, the second element specifying the number of data element columns, and the third element specifying the number of data element rows.
        CHW_3D     : The shape has three elements, with the first element specifying the number of channels or slices, the second element specifying the number of data element rows, and the third element specifying the number of data element columns.
    """
    UNSPECIFIED          =  0
    W_1D                 =  1
    H_1D                 =  2
    WH_2D                =  3
    HW_2D                =  4
    WHC_3D               =  5
    HWC_3D               =  6
    CWH_3D               =  7
    CHW_3D               =  8


SHAPE_ORDER_NAMES : List[str] = ['1d', 'w', 'h', '2d', 'wh', 'hw', '3d', 'whc', 'hwc', 'cwh', 'chw']
SHAPE_ORDER_LOOKUP: Dict[str, ShapeOrder] = {
    SHAPE_ORDER_NAMES[ 0]: ShapeOrder.UNSPECIFIED,
    SHAPE_ORDER_NAMES[ 1]: ShapeOrder.W_1D,
    SHAPE_ORDER_NAMES[ 2]: ShapeOrder.H_1D,
    SHAPE_ORDER_NAMES[ 3]: ShapeOrder.WH_2D,
    SHAPE_ORDER_NAMES[ 4]: ShapeOrder.WH_2D,
    SHAPE_ORDER_NAMES[ 5]: ShapeOrder.HW_2D,
    SHAPE_ORDER_NAMES[ 6]: ShapeOrder.WHC_3D,
    SHAPE_ORDER_NAMES[ 7]: ShapeOrder.WHC_3D,
    SHAPE_ORDER_NAMES[ 8]: ShapeOrder.HWC_3D,
    SHAPE_ORDER_NAMES[ 9]: ShapeOrder.CWH_3D,
    SHAPE_ORDER_NAMES[10]: ShapeOrder.CHW_3D
}

def parse_shape_order(value: str, default: ShapeOrder=ShapeOrder.UNSPECIFIED, raise_error: bool=False) -> ShapeOrder:
    """
    Attempt to parse a string value into a member of the `ShapeOrder` enumeration.

    Parameters
    ----------
        value      : The `str` value to parse.
        default    : The value to return if `value` is not one of the values in `radar.SHAPE_ORDER_NAMES`.
        raise_error: Specify `True` to raise a `ValueError` if `value` is not one of the values in `radar.SHAPE_ORDER_NAMES`.

    Returns
    -------
        One of the values of the `ShapeOrder` enumeration, or `default` if `value` is not known.
    """
    if not value:
        if raise_error:
            raise ValueError('The value argument is None, or is an empty string')
        else:
            return default

    enum_value: Optional[ShapeOrder] = SHAPE_ORDER_LOOKUP.get(value.lower(), None)
    if enum_value is not None:
        return enum_value
    elif raise_error:
        raise ValueError(f'The value argument \'{value}\' is not one of the recognized values {SHAPE_ORDER_NAMES}')
    else:
        return default


def shape_order_to_str(order: Union[int, ShapeOrder]) -> str:
    """
    Convert a value of the `ShapeOrder` enumeration to a descriptive string.

    Parameters
    ----------
        order: One of the values of the `ShapeOrder` enumeration.

    Returns
    -------
        A `str` description of the shape.
    """
    if order == ShapeOrder.UNSPECIFIED:
        return 'UNSPECIFIED'
    elif order == ShapeOrder.W_1D:
        return 'W'
    elif order == ShapeOrder.H_1D:
        return 'H'
    elif order == ShapeOrder.WH_2D:
        return 'WH'
    elif order == ShapeOrder.HW_2D:
        return 'HW'
    elif order == ShapeOrder.WHC_3D:
        return 'WHC'
    elif order == ShapeOrder.HWC_3D:
        return 'HWC'
    elif order == ShapeOrder.CWH_3D:
        return 'CWH'
    elif order == ShapeOrder.CHW_3D:
        return 'CHW'
    else:
        return f'UNKNOWN {int(order)}'


def indices_for_shape_order(order: Union[int, ShapeOrder], base_index: int=0, raise_error: bool=True) -> Tuple[int, int, int]:
    """
    Given a value of the `ShapeOrder` enumeration, return a tuple specifying the indices into a shape tuple.

    Parameters
    ----------
        order      : One of the values of the `ShapeOrder` enumeration, specifying the meaning of elements within the shape tuple.
        base_index : An integer value to add to the returned indices.
        raise_error: Specify `True` to raise a `ValueError` if `order` specifies an invalid `ShapeOrder` value.

    Returns
    -------
        A `tuple(int, int, int)` where:
        * The first element is the index of the width dimension, or `None`,
        * The second element is the index of the height dimension, or `None`,
        * The third element is the index of the depth dimension (number of channels, number of slices, etc.), or `None`.
        If the input `order` is invalid, the tuple `(None, None, None)` is returned.
    """
    if order == ShapeOrder.UNSPECIFIED:
        return (0 + base_index, None, None)
    elif order == ShapeOrder.W_1D:
        return (0 + base_index, None, None)
    elif order == ShapeOrder.H_1D:
        return (None, 0 + base_index, None)
    elif order == ShapeOrder.WH_2D:
        return (0 + base_index, 1 + base_index, None)
    elif order == ShapeOrder.HW_2D:
        return (1 + base_index, 0 + base_index, None)
    elif order == ShapeOrder.WHC_3D:
        return (0 + base_index, 1 + base_index, 2 + base_index)
    elif order == ShapeOrder.HWC_3D:
        return (1 + base_index, 0 + base_index, 2 + base_index)
    elif order == ShapeOrder.CWH_3D:
        return (1 + base_index, 2 + base_index, 0 + base_index)
    elif order == ShapeOrder.CHW_3D:
        return (2 + base_index, 1 + base_index, 0 + base_index)
    elif raise_error:
        raise ValueError(f'The value {int(order)} of the order argument is not a valid value of the ShapeOrder enumeration')
    else:
        return (None, None, None)


class StorageLayout(IntEnum):
    """
    Define identifiers for the supported data storage layouts.

    Fields
    ------
        CONTIGUOUS : The default layout. Array data is stored contiguously as [array1][array2][array3]...[arrayN]. When in doubt, store in `CONTIGUOUS` format.
        INTERLEAVED: Data is stored interleaved for increased compressibility, but is slower to load into memory, for example [array1_row1][array2_row1][array1_row2][array2_row2]...[array1_rowN][array2_rowN].
    """
    CONTIGUOUS  =  0
    INTERLEAVED =  1


STORAGE_LAYOUT_NAMES : List[str] = ['contiguous', 'c', 'interleaved', 'i']
STORAGE_LAYOUT_LOOKUP: Dict[str, StorageLayout] = {
    STORAGE_LAYOUT_NAMES[ 0]: StorageLayout.CONTIGUOUS,
    STORAGE_LAYOUT_NAMES[ 1]: StorageLayout.CONTIGUOUS,
    STORAGE_LAYOUT_NAMES[ 2]: StorageLayout.INTERLEAVED,
    STORAGE_LAYOUT_NAMES[ 3]: StorageLayout.INTERLEAVED
}


def parse_storage_layout(value: str, default: StorageLayout=StorageLayout.CONTIGUOUS, raise_error: bool=False) -> StorageLayout:
    """
    Attempt to parse a string value into a member of the `StorageLayout` enumeration.

    Parameters
    ----------
        value      : The `str` value to parse.
        default    : The value to return if `value` is not one of the values in `radar.STORAGE_LAYOUT_NAMES`.
        raise_error: Specify `True` to raise a `ValueError` if `value` is not one of the values in `radar.STORAGE_LAYOUT_NAMES`.

    Returns
    -------
        One of the values of the `StorageLayout` enumeration, or `default` if `value` is not known.
    """
    if not value:
        if raise_error:
            raise ValueError('The value argument is None, or is an empty string')
        else:
            return default

    enum_value: Optional[StorageLayout] = STORAGE_LAYOUT_LOOKUP.get(value.lower(), None)
    if enum_value is not None:
        return enum_value
    elif raise_error:
        raise ValueError(f'The value argument \'{value}\' is not one of the recognized values {STORAGE_LAYOUT_NAMES}')
    else:
        return default


def storage_layout_to_str(layout: Union[int, StorageLayout]) -> str:
    """
    Convert a value of the `StorageLayout` enumeration to a descriptive string.

    Parameters
    ----------
        layout: One of the values of the `StorageLayout` enumeration.

    Returns
    -------
        A `str` description of the layout.
    """
    if layout == StorageLayout.CONTIGUOUS:
        return 'CONTIGUOUS'
    elif layout == StorageLayout.INTERLEAVED:
        return 'INTERLEAVED'
    else:
        return f'UNKNOWN {int(layout)}'


def dtype_to_data_format(dtype, raise_error: bool=True) -> DataFormat:
    """
    Convert a `numpy dtype` to the corresponding value of the `DataFormat` enumeration.

    Parameters
    ----------
        dtype      : The `numpy dtype` value.
        raise_error: Specify `True` to raise a `ValueError` if `dtype` is `None` or is an unsupported type.

    Returns
    -------
        One of the values of the `DataFormat` enumeration, or `DataFormat.UNKNOWN` if `dtype` is unsupported.
    """
    if dtype is None:
        if raise_error:
            raise ValueError('The supplied dtype argument is None')
        else:
            return DataFormat.UNKNOWN

    if dtype == np.int8:
        return DataFormat.S8
    elif dtype == np.uint8:
        return DataFormat.U8
    elif dtype == np.int16:
        return DataFormat.S16
    elif dtype == np.uint16:
        return DataFormat.U16
    elif dtype == np.int32:
        return DataFormat.S32
    elif dtype == np.uint32:
        return DataFormat.U32
    elif dtype == np.int64:
        return DataFormat.S64
    elif dtype == np.uint64:
        return DataFormat.U64
    elif dtype == np.float16:
        return DataFormat.F16
    elif dtype == np.float32:
        return DataFormat.F32
    elif dtype == np.float64:
        return DataFormat.F64
    else:
        if raise_error:
            raise ValueError(f'The specfied dtype {dtype} is not supported')
        else:
            return DataFormat.UNKNOWN


def data_format_to_dtype(format: Union[int, DataFormat], raise_error: bool=True) -> Optional[np.dtype]:
    """
    Convert a value of the `DataFormat` enumeration into an `numpy dtype` instance.

    Parameters
    ----------
        format     : The `DataFormat` value to convert.
        raise_error: Specify `True` to raise a `ValueError` if `format` cannot be converted into a `numpy dtype`.

    Returns
    -------
        The corresponding `numpy dtype`, or `None` if an error occurred.
    """
    if format is None:
        if raise_error:
            raise ValueError('The supplied format argument is None')
        else:
            return None

    if format == DataFormat.S8:
        return np.dtype(np.int8)
    elif format == DataFormat.U8:
        return np.dtype(np.uint8)
    elif format == DataFormat.S16:
        return np.dtype(np.int16)
    elif format == DataFormat.U16:
        return np.dtype(np.uint16)
    elif format == DataFormat.S32:
        return np.dtype(np.int32)
    elif format == DataFormat.U32:
        return np.dtype(np.uint32)
    elif format == DataFormat.S64:
        return np.dtype(np.int64)
    elif format == DataFormat.U64:
        return np.dtype(np.uint64)
    elif format == DataFormat.F16:
        return np.dtype(np.float16)
    elif format == DataFormat.F32:
        return np.dtype(np.float32)
    elif format == DataFormat.F64:
        return np.dtype(np.float64)
    else:
        if raise_error:
            raise ValueError(f'The supplied format argument {int(format)} is unsupported')
        else:
            return 0


def data_component_size(format: Union[int, DataFormat], raise_error: bool=True) -> int:
    """
    Calculate the number of bytes needed to store a single component with a given `DataFormat`.

    Parameters
    ----------
        format     : The `DataFormat` used to represent the component value.
        raise_error: Specify `True` to raise a `ValueError` if `format` is `DataFormat.UNKNOWN` or some invalid value.

    Returns
    -------
        The size of a single component, in bytes, or 0 if `format` is `DataFormat.UNKNOWN` or some invalid value.
    """
    if format == DataFormat.UNKNOWN:
        if raise_error:
            raise ValueError('Cannot determine component size for UNKNOWN format')
        else:
            return 0
    elif format == DataFormat.S8:
        return 1
    elif format == DataFormat.U8:
        return 1
    elif format == DataFormat.S16:
        return 2
    elif format == DataFormat.U16:
        return 2
    elif format == DataFormat.F16:
        return 2
    elif format == DataFormat.S32:
        return 4
    elif format == DataFormat.U32:
        return 4
    elif format == DataFormat.F32:
        return 4
    elif format == DataFormat.S64:
        return 8
    elif format == DataFormat.U64:
        return 8
    elif format == DataFormat.F64:
        return 8
    else:
        if raise_error:
            raise ValueError(f'Cannot determine component size for unsupported DataFormat value {format}')
        else:
            return 0


def data_element_size(format: Union[int, DataFormat], interpret: Union[int, DataInterpretation], raise_error: bool=True) -> int:
    """
    Calculate the total size of a single data element.

    Parameters
    ----------
        format     : One of the values of the `DataFormat` enumeration, specifying the storage format for each component in the data element.
        interpret  : One of the values of the `DataInterpretation` enumeration, specifying the number of components in the data element.
        raise_error: Specify `True` to raise a `ValueError` if the `format` or `interpret` arguments specify an invalid value.

    Returns
    -------
        The size of a single data element, in bytes, or 0 if an error occurred.
    """
    component_size : int = data_component_size(format, raise_error)
    component_count: int = 0

    if interpret == DataInterpretation.UNSPECIFIED:
        component_count = 1
    elif interpret == DataInterpretation.SCALAR:
        component_count = 1
    elif interpret == DataInterpretation.VEC2:
        component_count = 2
    elif interpret == DataInterpretation.VEC3:
        component_count = 3
    elif interpret == DataInterpretation.RGB:
        component_count = 3
    elif interpret == DataInterpretation.VEC4:
        component_count = 4
    elif interpret == DataInterpretation.RGBX:
        component_count = 4
    elif interpret == DataInterpretation.RGBA:
        component_count = 4
    else:
        if raise_error:
            raise ValueError(f'Cannot determine component count for unsupported DataInterpretation value {int(interpret)}')
        else:
            component_count = 0

    return component_count * component_size


def shape_length_for_shape_order(order: Union[int, ShapeOrder], raise_error: bool=True) -> int:
    """
    Determine the number of elements expected in a shape tuple for a given `ShapeOrder` value.

    Parameters
    ----------
        order      : One of the values of the `ShapeOrder` enumeration.
        raise_error: Specify `True` to raise a `ValueError` if the `order` argument is invalid.

    Returns
    -------
        The expected length of the shape tuple. This will be either 1, 2, or 3. The return value is 0 if `order` specifies an unknown value.
    """
    if order == ShapeOrder.UNSPECIFIED:
        return 1
    elif order == ShapeOrder.W_1D:
        return 1
    elif order == ShapeOrder.H_1D:
        return 1
    elif order == ShapeOrder.WH_2D:
        return 2
    elif order == ShapeOrder.HW_2D:
        return 2
    elif order == ShapeOrder.WHC_3D:
        return 3
    elif order == ShapeOrder.HWC_3D:
        return 3
    elif order == ShapeOrder.CWH_3D:
        return 3
    elif order == ShapeOrder.CHW_3D:
        return 3
    else:
        if raise_error:
            raise ValueError(f'Cannot determine shape tuple length for unsupported ShapeOrder value {int(order)}')
        else:
            return 0


def array_element_count(shape: Optional[Tuple[int, ...]]) -> int:
    """
    Calculate the number of data elements in an array given the shape of the array.

    Parameters
    ----------
        shape: A `tuple` of `int` specifying the shape of the array.

    Returns
    -------
        The total number of array elements.
    """
    if shape is None or len(shape) == 0:
        return 0

    count: int = shape[0]
    for index in range(1, len(shape)):
        count *= shape[index]

    return count


def _validate_shape(order: Union[int, ShapeOrder], shape: Optional[Tuple[int, ...]]=None, raise_error: bool=True) -> bool:
    """
    Validate an array shape value against limits defined by the data format.

    Parameters
    ----------
        order      : One of the values of the `ShapeOrder` enumeration, specifying how the shape is interpreted and the expected shape dimension.
        shape      : The array shape to validate.
        raise_error: Specify `True` to raise a `ValueError` for any constraint violation.

    Returns
    -------
        `True` if `shape` appears to be valid, or `False` if `shape` is `None` or otherwise invalid.
    """
    if shape is None:
        return False

    expected_length: int = shape_length_for_shape_order(order, raise_error)
    if expected_length == 0:
        return False

    if len(shape) != expected_length:
        if raise_error:
            raise ValueError(f'The length of the array shape, {len(shape)}, does not match length {expected_length} indicated by ShapeOrder {shape_order_to_str(order)}')
        else:
            return False

    max_dim: int = MAXIMUM_DIMENSION
    for idx, dim in enumerate(shape):
        if dim > max_dim:
            if raise_error:
                raise ValueError(f'Array shape dimension {idx} value {dim} exceeds the maximum dimension {max_dim}')
            else:
                return False

    return True


class RawArrayDataHeaderV1:
    """
    Provides easy access to the fields loaded from a Raw Array data header, version 1.

    Fields
    ------
    """
    def __init__(self, order= ShapeOrder.UNSPECIFIED, format=DataFormat.UNKNOWN, interpret=DataInterpretation.UNSPECIFIED, storage=StorageLayout.CONTIGUOUS) -> None:
        self.magic          : bytes              = MAGIC_BYTES
        self.version        : int                = 1
        self.storage_layout : StorageLayout      = storage
        self.shape_order    : ShapeOrder         = order
        self.data_format    : DataFormat         = format
        self.interpretation : DataInterpretation = interpret
        self.element_stride : int                = 0
        self.x_stride       : int                = 0
        self.array_stride   : int                = 0
        self.array_count    : int                = 0
        self.array_shape    : Tuple[int, ...]    = None


def make_header_v1(format: Union[int, DataFormat], interpret: Union[int, DataInterpretation], storage: Union[int, StorageLayout], shape_order: Union[int, ShapeOrder], array_shape: Tuple[int, ...], array_count: int) -> Tuple[RawArrayDataHeaderV1, bytes]:
    """
    Create a `bytes` object containing a binary-serialized Raw Array Data header, version 1.

    Parameters
    ----------
        format     : One of the values of the `DataFormat` enumeration, specifying the data format of the individual data elements. This value may not be `DataFormat.UNKNOWN`.
        interpret  : One of the values of the `DataInterpretation` enumeration, specifying how the data in each array should be interpreted.
        storage    : One of the values of the `StorageLayout` enumeration, specifying the storage format for the array data.
        shape_order: One of the values of the `ShapeOrder` enumeration, specifying the length of the shape tuple and indicating how each element of the shape tuple should be interpreted.
        array_shape: A `tuple` of `int` values specifying the array dimension(s). The length of this `tuple` should match the expected length given by `shape_order`.
        array_count: The number of distinct data arrays stored in the data record. This value may be 0.

    Returns
    -------
        A `tuple(RawArrayDataHeaderV1, bytes)` where:
        * The first element is a `RawArrayDataHeaderV1` instance storing the specified values along with any derived values, such as byte strides,
        * The second element is a `bytes` instance of exactly `radar.HEADER_SIZE_V1` bytes containing the serialized header data.
        
    Exceptions
    ----------
        ValueError         : Raised if the value of the `format`, `interpret`, `storage`, `shape_order`, `array_shape` or `array_count` arguments is invalid.
        NotImplementedError: Raised if the value of `storage` is `StorageLayout.INTERLEAVED`.
    """
    order_int    : int = int(shape_order)
    order_min    : int = int(ShapeOrder.UNSPECIFIED)
    order_max    : int = int(ShapeOrder.CHW_3D)
    format_int   : int = int(format)
    format_min   : int = int(DataFormat.UNKNOWN)
    format_max   : int = int(DataFormat.F64)
    storage_int  : int = int(storage)
    storage_min  : int = int(StorageLayout.CONTIGUOUS)
    storage_max  : int = int(StorageLayout.INTERLEAVED)
    interpret_int: int = int(interpret)
    interpret_min: int = int(DataInterpretation.UNSPECIFIED)
    interpret_max: int = int(DataInterpretation.RGBA)

    if format_int == int(DataFormat.UNKNOWN):
        raise ValueError('The format argument cannot be DataFormat.UNKNOWN')
    if format_int < format_min or format_int > format_max:
        raise ValueError(f'Value {format_int} specified for format argument is out of range [{format_min}, {format_max}]')
    if storage_int == int(StorageLayout.INTERLEAVED):
        raise NotImplementedError('Support for INTERLEAVED storage is not yet implemented')
    if storage_int < storage_min or storage_int > storage_max:
        raise ValueError(f'Value {storage_int} specified for storage argument is out of range [{storage_min}, {storage_max}]')
    if interpret_int < interpret_min or interpret_int > interpret_max:
        raise ValueError(f'Value {interpret_int} specified for interpret argument is out of range [{interpret_min}, {interpret_max}]')
    if order_int < order_min or order_int > order_max:
        raise ValueError(f'Value {order_int} specified for shape_order argument is out of range [{order_min}, {order_max}]')
    if array_shape is None:
        raise ValueError('The array_shape argument must be specified')
    if array_count < 0 or array_count > MAXIMUM_DIMENSION:
        raise ValueError(f'Value {array_count} specified for the array_count argument is out of range [0, {MAXIMUM_DIMENSION}]')

    _validate_shape(shape_order, array_shape, raise_error=True)
    element_count = array_element_count(array_shape)

    # Compute the individual values to return to the caller.
    header_v1 = RawArrayDataHeaderV1(shape_order, format, interpret, storage)
    header_v1.element_stride = data_element_size (format, interpret, raise_error=True)
    header_v1.x_stride       = header_v1.element_stride * array_shape[0]
    header_v1.array_stride   = header_v1.element_stride * element_count
    header_v1.array_count    = array_count
    header_v1.array_shape    = tuple(array_shape)

    # The header structure is a fixed size, and any unused bytes are set to 0.
    offset  = 0
    VERSION = 1
    header  = bytearray(HEADER_SIZE_V1)
    struct.pack_into("<4sB", header, offset, MAGIC_BYTES, VERSION)
    offset += 5 # offset -> 5
    struct.pack_into("<5B" , header, offset, storage_int, order_int, format_int, interpret_int, header_v1.element_stride)
    offset += 5 # offset -> 10
    struct.pack_into("<I"  , header, offset, header_v1.x_stride)
    offset += 4 # offset -> 14
    struct.pack_into("<Q"  , header, offset, header_v1.array_stride)
    offset += 8 # offset -> 22
    struct.pack_into("<H"  , header, offset, header_v1.array_count)
    offset += 2 # offset -> 24
    for index in range(element_count):
        struct.pack_into("<H", header, offset, array_shape[index])
        offset += 2 # offset -> 26/28/30

    return (header_v1, header)


def load_header_v1(src: Union[ByteString, io.BufferedIOBase]) -> Tuple[Optional[RawArrayDataHeaderV1], Optional[bytes]]: 
    """
    Load a Raw Array DAta Record header, version 1, from a byte stream.
    Note that the version field of the header is NOT validated to contain the value '1'. Any value greater than or equal to 1 is accepted.
    This operation consumes up to `radar.HEADER_SIZE_V1` bytes from the source stream.

    Parameters
    ----------
        src: The byte stream or binary file object to read from.

    Returns
    -------
        A `tuple(RawArrayDataHeaderV1, bytes)` where:
        * The first element is a `RawArrayDataHeaderV1` instance that can be used to access the individual header fields,
        * The second element is a `bytes` object of length `radar.HEADER_SIZE_V1` containing the raw header data.
        If an error occurs, the return value is `(None, None)` and the source stream read position is reset to the read pointer position of the stream when the function was initially called.

    Exceptions
    ----------
        ValueError: Raised if the value of the `src` argument is `None`, or the shape order specified in the header is invalid.
    """
    if src is None:
        raise ValueError('The src argument is None')

    stream: io.BufferedIOBase = src
    if not isinstance(src, io.BufferedIOBase):
        stream = io.BytesIO(stream)

    base  : int   = stream.tell()
    offset: int   = 0
    header: bytes = stream.read(HEADER_SIZE_V1)
    if header is None or len(header) != HEADER_SIZE_V1:
        stream.seek(base, os.SEEK_SET)
        return (None, None)

    magic: bytes = header[0:4]
    version: int = header[4:5]
    if magic != MAGIC_BYTES or version < 1:
        stream.seek(base, os.SEEK_SET)
        return (None, None)

    # Read the remaining v1 header fields.
    offset   = 5 # offset -> 5
    storage, order, format, interpret, element_stride = struct.unpack_from("<5B", header, offset)
    offset  += 5 # offset -> 10
    x_stride = struct.unpack_from("<I", header, offset)[0]
    offset  += 4 # offset -> 14
    a_stride = struct.unpack_from("<Q", header, offset)[0]
    offset  += 8 # offset -> 22
    a_count  = struct.unpack_from("<H", header, offset)[0]
    offset  += 2 # offset -> 24
    a_length = shape_length_for_shape_order(order, raise_error=True)
    a_shape  = struct.unpack_from(f'<{a_length}H', header, offset)
    offset  += 2 * a_length # offset -> 26/28/30

    # Construct the header object.
    header_v1 = RawArrayDataHeaderV1(order, format, interpret, storage)
    header_v1.version        = version # May not actually be 1
    header_v1.element_stride = data_element_size(format, interpret, raise_error=True)
    header_v1.x_stride       = x_stride
    header_v1.array_stride   = a_stride
    header_v1.array_count    = a_count
    header_v1.array_shape    = a_shape
    return (header_v1, header)


def append_array(dst: io.BufferedIOBase, src: np.ndarray, header: RawArrayDataHeaderV1, dtype: Optional[np.dtype]=None) -> int:
    """
    Append an array object to a Raw Array DAta Record binary data stream.
    This high-level routine will perform data format conversions and reshaping as necessary.

    Parameters
    ----------
        dst   : The destination binary data stream. The write pointer should be positioned at the offset where the array data will be written.
        src   : The `numpy` array to append to the data stream.
        header: The header specifying the basic expected attributes of the array. The array specified by the `src` argument is validated against the header attributes.
        dtype : The `numpy dtype` corresponding to the `DataFormat` specified in the header, or `None` to determine the `dtype` based on the array dtype.

    Returns
    -------
        The number of bytes written to the destination data stream, or 0 if an error occurred.

    Exceptions
    ----------
        ValueError: Raised if `dst`, `src` or `header` is `None`, or if the attributes of `src` do not match the attributes specified in `header`.
    """
    if dst is None:
        raise ValueError('The dst argument cannot be None')
    if src is None:
        raise ValueError('The src argument cannot be None')
    if header is None:
        raise ValueError('The header argument cannot be None')
    if dtype is None:
        dtype = data_format_to_dtype(header.data_format, raise_error=True)

    # Determine if any reshaping is needed.
    src_elm = array_element_count(src.shape)
    dst_elm = array_element_count(header.array_shape)
    if src_elm != dst_elm:
        raise ValueError(f'The number of data elements {src_elm} in the source array does not match the number of data elements {dst_elm} specified in the header')
    if src.shape != header.array_shape:
        src = src.reshape(header.array_shape)

    # Perform a format conversion for the data elements, if necessary.
    if src.dtype != dtype:
        src = src.astype(dtype)

    # Convert to a raw byte array and write everything to the stream.
    array_bytes: bytes = src.tobytes(order='C')
    return dst.write(array_bytes)


def append_bytes(dst: io.BufferedIOBase, src: ByteString, header: RawArrayDataHeaderV1) -> int:
    """
    Append a raw byte array to a Raw Array DAta Record binary data stream.
    The size of the `src` buffer must match the `array_stride` specified in `header`.

    Parameters
    ----------
        dst   : The destination binary data stream. The write pointer should be positioned at the offset where the data will be written.
        src   : The source `bytes-like` object specifying the data to write. The length of this buffer must match the `array_stride` specified in the `header` argument.
        header: The header specifying the basic expected attributes of the arrays stored in the data record.

    Returns
    -------
        The number of bytes written to the destination data stream, or 0 if an error occurred.

    Exceptions
    ----------
        ValueError: Raised if `dst`, `src` or `header` is `None`, or if the attributes of `src` do not match the attributes specified in `header`.
    """
    if dst is None:
        raise ValueError('The dst argument cannot be None')
    if src is None:
        raise ValueError('The src argument cannot be None')
    if header is None:
        raise ValueError('The header argument cannot be None')
    if len(src) != header.array_stride:
        raise ValueError(f'The length of the source buffer {len(src)} does not match the array stride {header.array_stride}')

    return dst.write(src)


def load_array(src: io.BufferedIOBase, header: RawArrayDataHeaderV1, offset: int=-1, dtype: Optional[np.dtype]=None) -> Tuple[int, Optional[np.ndarray]]:
    """
    Load an array from a Raw Array DAta Record binary data stream into a new `numpy` array.

    Parameters
    ----------
        src   : The source binary data stream.
        header: The header specifying the attributes of the array(s) stored in the data stream.
        offset: The byte offset, relative to the start of the source data stream, at which to begin reading array data. If this value is -1, the read is performed starting at the current read pointer within the source stream.
        dtype : The `numpy dtype` of the array. This must match the `dtype` associated with the `DataFormat` specified in `header.data_format`. If this value is `None`, the `dtype` is determined from the `header.data_format` field.

    Returns
    -------
        A `tuple(int, ndarray)` where:
        * The first element is the number of bytes read from the source data stream, which should correspond to `header.array_stride`,
        * The second element is a `numpy ndarray` object initialized with the data read from the source data stream.
        If an error occurs, the return value is `(0, None)`.

    Exceptions
    ----------
        ValueError: Raised if either the `src` or `header` arguments are `None`, or if the `dtype` cannot be determined.
    """
    if src is None:
        raise ValueError('The src argument cannot be None')
    if header is None:
        raise ValueError('The header argument cannot be None')
    if header.array_stride == 0:
        raise ValueError('The header specifies an array_stride of 0')
    if dtype is None:
        dtype = data_format_to_dtype(header.data_format, raise_error=True)

    base: int = src.tell()
    if offset == -1:
        offset = base

    # Seek to the given offset and read the raw data.
    dst = bytearray(header.array_stride)
    src.seek(offset, os.SEEK_SET)
    nread = src.readinto(dst)
    if nread != header.array_stride:
        src.seek(base, os.SEEK_SET)
        return (0, None)

    # Create a new numpy array that uses the underlying buffer.
    result_array = np.frombuffer(dst, dtype)
    result_array = result_array.reshape(header.array_shape)
    return (nread, result_array)


def load_array_into(dst: memoryview, src: io.BufferedIOBase, header: RawArrayDataHeaderV1, offset: int=-1, dtype: Optional[np.dtype]=None) -> int:
    """
    Load an array from a Raw Array DAta Record binary data stream into an existing buffer.

    Parameters
    ----------
        dst   : The destination buffer.
        src   : The source binary data stream.
        header: The header specifying the attributes of the array(s) stored in the data stream.
        offset: The byte offset, relative to the start of the source data stream, at which to begin reading array data. If this value is -1, the read is performed starting at the current read pointer within the source stream.
        dtype : The `numpy dtype` of the array. This must match the `dtype` associated with the `DataFormat` specified in `header.data_format`. If this value is `None`, the `dtype` is determined from the `header.data_format` field.

    Returns
    -------
        The number of bytes loaded into the destination buffer, which will be either `header.array_stride or 0 if an error occurred.

    Exceptions
    ----------
        ValueError: Raised if any of the `dst`, `src` or `header` arguments are `None`, or if `dst` is smaller than `header.array_stride`.
    """
    if dst is None:
        raise ValueError('The dst argument cannot be None')
    if src is None:
        raise ValueError('The src argument cannot be None')
    if header is None:
        raise ValueError('The header argument cannot be None')
    if header.array_stride == 0:
        raise ValueError('The header specifies an array_stride of 0')
    if dst.nbytes != header.array_stride:
        raise ValueError(f'The size of the destination buffer {dst.nbytes} does not match the array_stride {header.array_stride}')
    if dtype is None:
        dtype = data_format_to_dtype(header.data_format, raise_error=True)

    base: int = src.tell()
    if offset == -1:
        offset = base

    # Seek to the given offset and read the raw data.
    src.seek(offset, os.SEEK_SET)
    nread = src.readinto(dst)
    if nread != header.array_stride:
        src.seek(base, os.SEEK_SET)
        return 0

    return nread
