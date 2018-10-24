"""
Module used for reading of htk files.
"""

"""
This file is originally based on sources available in BRAINformat. The file has been
modified to meet NWB needs.


BRAINFormat Copyright (c) 2014, 2015, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required approvals
from the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this list
    of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright notice, this
    list of conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley National
    Laboratory, U.S. Dept. of Energy nor the names of its contributors may be
    used to endorse or promote products derived from this software without specific
    prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
to the features, functionality or performance of the source code ("Enhancements")
to anyone; however, if you choose to make your Enhancements available either publicly,
or directly to Lawrence Berkeley National Laboratory, without imposing a separate
written license agreement for such Enhancements, then you hereby grant the following
license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software, distribute, and
sublicense such enhancements or derivative works thereof, in binary and source code form.
"""

from struct import unpack
import numpy as np
import sys


class HTKFormat(object):
    """
    Specification of base information about the HTK file format.
    """

    param_kind_base = {
        "WAVEFORM": 0,   # sampled waveform
        "LPC": 1,        # linear prediction filter coefficients
        "LPCREFC": 2,    # linear prediction reflection coefficients
        "LPCEPSTRA": 3,  # LPC cepstral coefficients
        "LPCDELCEP": 4,  # LPC cepstra plus delta coefficients
        "IREFC": 5,      # LPC reflection coefficient in 16 bit integer format
        "MFCC": 6,       # mel-frequency cepstral coefficients
        "FBANK": 7,      # log mel-filter bank channel outputs
        "MELSPEC": 8,    # linear mel-filter bank channel outputs
        "USER": 9,       # user-defined sample kind
        "DISCRETE": 10,  # vector quantised data
    }
    """
    Dictionary describing the basic parameter kind codes.

        * WAVEFORM = 0   : sampled waveform
        * LPC = 1        : linear prediction filter coefficients
        * LPCREFC = 2    : linear prediction reflection coefficients
        * LPCEPSTRA = 3  : LPC cepstral coefficients
        * LPCDELCEP = 4  : LPC cepstra plus delta coefficients
        * IREFC = 5      : LPC reflection coefficient in 16 bit integer format
        * MFCC = 6       : mel-frequency cepstral coefficients
        * FBANK = 7      : log mel-filter bank channel outputs
        * MELSPEC = 8    : linear mel-filter bank channel outputs
        * USER = 9       : user-defined sample kind
        * DISCRETE = 10  : vector quantised data
    """

    param_kind_encoding = {
        "_E": 0o000100,  # has energy
        "_N": 0o000200,  # absolute energy suppressed
        "_D": 0o000400,  # has delta coefficients
        "_A": 0o001000,  # has acceleration (delta-delta) coefficients
        "_C": 0o002000,  # is compressed
        "_Z": 0o004000,  # has zero mean static coefficients
        "_K": 0o010000,  # has CRC checksum
        "_O": 0o020000,  # has 0th cepstral coefficient
    }
    """
    Dictionary describing the parameter kind encodings.

        * _E = 0000100 : has energy
        * _N = 0000200 : absolute energy suppressed
        * _D = 0000400 : has delta coefficients
        * _A = 0001000 : has acceleration (delta-delta) coefficients
        * _C = 0002000 : is compressed
        * _Z = 0004000 : has zero mean static coefficients
        * _K = 0010000 : has CRC checksum
        * _O = 0020000 : has 0th cepstral coefficient
    """

    header = [
        {'name': 'num_samples', 'format': 'I', 'description': 'Number of samples in the File'},
        {'name': 'sample_period', 'format': 'I', 'description': 'Sample period in 100ns units'},
        {'name': 'sample_size', 'format': 'H', 'description': 'Number of bytes per sample'},
        {'name': 'parameter_kind', 'format': 'H', 'description': 'A code indicating the sample kind'}
    ]
    """
    List describing the contents of the HTK file header.
    """

    header_length = 12
    """
    Total length in bytes of the file header.
    """

    byte_order = '>'
    """
    Byte-order in which the HTK data is written
    """

    @classmethod
    def header_format(cls):
        """
        Get the format string to unpack the header of the HTK file.

        :returns string---e.g. '>IIHH'---describing the format to be used for unpacking the header.

        """
        hf = cls.byte_order
        for he in cls.header:
            hf += he['format']
        return hf


class HTKFile(object):
    """
    Class used for reading HTK format files.

    NOTE: The original HTK specification specifies that the sample_period is given in the header in 100ns
    units. In some cases however, users appear to write the sampling rate in the header with a different
    base. We therefore allow users to specify the base for the sampling rate and if given we assume that
    the header contains the sampling rate and we convert accordingly.

    Instance Variables:

    :ivar filename: Name of the HTK file
    :ivar data: Numpy array of the data or None in case that read_data has not been called yet
    :ivar num_samples: Number of samples in the file
    :ivar sample_period: Sample period in 100ns units
    :ivar sample_rate: Sampling rate in Hz. This is the same as 10000/sample_period.
    :ivar sample_size: Number of bytes per sample
    :ivar parameter_kind: Code indicating the sample kind (see HTKFormat for details on parmKind)
    :ivar dtype: Data type
    :ivar vector_length: Vector length
    :ivar A: Compression parameter
    :ivar B: Compression parameter
    :ivar header_length: Total header length

    Internal Variables:

    :ivar __file: The handle to the HTK file
    :ivar __current_pos: Internal variable used to store the current sample position during iteration

    """
    def __init__(self,
                 filename,
                 sample_rate_base=10000.):
        """
        Initialize the htk file

        :param filename: The name of the htk file to be read
        :param sample_rate_base: None if the sample_period is given in the header. Set to the number that
            should we should divide the sampling rate given in the header by in order to convert the
            rate to the appropriate value in Hz.

        :return:
        """
        self.filename = filename
        self.__file = open(self.filename, 'rb')

        self.data = None  # Numpy array with all data or None
        self.__current_pos = 0  # Current sample position. This variable is used during iteration.

        # Read the HTK header to initialize the variables: self.num_samples,
        # self.sample_period, self.sample_size, self.parameter_kind
        # self.dtype, self.vector_length, self.A, self.B self.header_length

        # Position the file handle at the beginning of the file
        self.__file.seek(0, 0)
        # Read and unpack the header
        header_data = unpack(HTKFormat.header_format(),
                             self.__file.read(HTKFormat.header_length))
        # Save the header information
        self.num_samples = header_data[0]
        # The original HTK format specifies this to be the sample period in 100ns
        if not sample_rate_base:
            self.sample_period = header_data[1]
            self.sample_rate = (10000. / self.sample_period) * 1000.
        else:
            self.sample_rate = header_data[1] / float(sample_rate_base)
            self.sample_period = (1. / float(self.sample_rate)) * 1000000000

        self.sample_size = header_data[2]
        self.parameter_kind = header_data[3]

        # Get the coefficients for compressed data
        if self.parameter_kind & HTKFormat.param_kind_encoding['_C']:
            self.dtype = 'h'
            self.vector_length = self.sample_size / 2
            if self.parameter_kind & 0x3f == HTKFormat.param_kind_base['IREFC']:
                self.A = 32767
                self.B = 0
            else:
                self.A = np.fromfile(self.__file, 'f', self.vector_length)
                self.B = np.fromfile(self.__file, 'f', self.vector_length)
                if self.__swap_required():
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'
            self.vector_length = self.sample_size / 4
        self.header_length = self.__file.tell()

    def __iter__(self):
        """Make the HTKFile iterable"""
        self.__seek_sample(0)
        self.__current_pos = 0
        return self

    def __next__(self):
        """Get the next item for iteration"""
        if self.data is not None:
            d = self.data[self.__current_pos, :]
        else:
            d = self.read_sample(sample_index=self.__current_pos)
        self.__current_pos += 1
        return d

    @staticmethod
    def __is_big_endian():
        """
        Simple helper function used to determine whether the system is
        little or big endianess. This is needed to determine whether
        the binary data read needs to be swapped or not.

        :returns: Boolean indicating whether the data is big or little endian.
        """
        return sys.byteorder == 'big'

    def __swap_required(self):
        """
        Method use to check whether we need to swap the byteorder of the
        binary data read.

        :returns: Boolean indicating if the byteorder needs to be swapped.
        """
        return not self.__is_big_endian()

    def __seek_sample(self, sample_index=0):
        """
        Internal helper function used to position the file handle at the position
        of the sample with the given index.
        """
        target_position = self.header_length
        target_position += (sample_index * self.sample_size)
        self.__file.seek(target_position, 0)

    def read_sample(self, sample_index):
        """
        Read the data of a single sample with the given index.

        :param sample_index: The index of the sample to be read

        :returns: The vector with the data for the sample.
        """

        # If we have read all data then return the data directly
        if self.data is not None:
            return self.data[sample_index, :]
        # If the data has not been read yet, then read it from file
        else:
            # Put the file handler at the position of the sample
            self.__seek_sample(sample_index)
            # Read the data for the sample
            tempvec = np.fromfile(self.__file, self.dtype, int(self.vector_length))
            # Perform byteswap if necessary
            if self.__swap_required():
                tempvec = tempvec.byteswap()
            # Uncompress data to floats if needed
            if self.parameter_kind & HTKFormat.param_kind_encoding['_C']:
                tempvec = (tempvec.astype('f') + self.B) / self.A
            return tempvec

    def read_data(self):
        """
        Get a numpy data array of all the samples

        :returns: Numpy data array of all the samples
        """
        # Jump to the beginning of the file
        self.__seek_sample(0)
        # Read all data
        tempdata = np.fromfile(self.__file, self.dtype)
        # Remove the checksum data
        if self.parameter_kind & HTKFormat.param_kind_encoding['_K']:
            tempdata = tempdata[:-1]
        # Reshape the data to (#vectors, #vector_length
        outshape = (int(len(tempdata)/self.vector_length), int(self.vector_length))
        tempdata = tempdata.reshape(outshape)
        if self.__swap_required():
            tempdata = tempdata.byteswap()
        # Uncompress data to floats if needed
        if self.parameter_kind & HTKFormat.param_kind_encoding['_C']:
            tempdata = (tempdata.astype('f') + self.B) / self.A
        self.data = tempdata
        return self.data
