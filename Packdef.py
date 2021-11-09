import struct
import ctypes

sockets = []
pthreads = []
# 退出标志
quitFlag = True

# 服务器信息
service_IP = "127.0.0.1"
service_Port = 9090


DEF_ANALYSIS_BASE_RQ = 20000

DEF_ANALYSIS_FAIL_RQ = DEF_ANALYSIS_BASE_RQ - 1

DEF_ANALYSIS_QUIT_RQ = DEF_ANALYSIS_BASE_RQ + 0
DEF_ANALYSIS_CHATTER_RQ = DEF_ANALYSIS_BASE_RQ + 1
DEF_ANALYSIS_CHATTER_RS = DEF_ANALYSIS_BASE_RQ + 2

DEF_ANALYSIS_SPEECH_RQ = DEF_ANALYSIS_BASE_RQ + 3
DEF_ANALYSIS_BASEMOVE_RS = DEF_ANALYSIS_BASE_RQ + 4


# value1 = (1,b'good',1.22)
# s1 = struct.Struct('i10sf')
# values = (DEF_ANALYSIS_CHATTER_RS, b'command_analysis server state normal')
# s = struct.Struct('i100s')
# buff = ctypes.create_string_buffer(s.size)
# pack_date = s.pack_into(buff, 0, *values)
# buff2 = s.pack(*values)
# print(len(buff2))
# unpacked_date = s.unpack_from(buff,0)
# print(len(buff))
# print("origin values",value1)
# print("format string",s1.format)
# print("buffer ",buff)
# print("unpacked val",unpacked_date)