import struct
import ctypes

if __name__ == '__main__':
    value = (b"hello     ",)
    s = struct.Struct('10s')
    buff = s.pack(*value)

    unpack = s.unpack(buff)
    strval = str(unpack[0],encoding='utf-8')
    strclear = strval.rstrip()
    print(strclear)