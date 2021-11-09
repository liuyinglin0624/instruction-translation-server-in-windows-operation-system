import TCPNet
from Packdef import *
from command_analysis import *


class Kernel():
    def __init__(self):
        self.m_net = TCPNet.TCPNet(self)

    def initNet(self):
        self.m_net.InitNetWork()
        LoadInit()  # 加载模型

    def DealDate(self, socket, Date, nLen):
        # size_date[0]表示协议类型
        # size_date[1]表示协议内容
        if Date[0] == DEF_ANALYSIS_CHATTER_RQ:
            print(Date[1])
            values = (DEF_ANALYSIS_CHATTER_RS, b'command_analysis server state normal')
            s = struct.Struct('i100s')
            buff = s.pack(*values)
            self.SendDate(socket, buff, len(buff))
        elif Date[0] == DEF_ANALYSIS_SPEECH_RQ:
            strAnalysis = str(Date[1], encoding='utf-8')
            strClear = ''
            for i in range(len(strAnalysis)):
                if not strAnalysis[i] == strAnalysis[len(strAnalysis)-1]:
                    strClear += strAnalysis[i]

            retVal = GetInstructonKey(strClear)
            print(retVal)
            classval = retVal[0]
            res = retVal[1]
            if res is not None and len(res) != 0:
                # 基本运动类
                if classval == 1:
                    values = None
                    if len(res) == 2:
                        values = (DEF_ANALYSIS_BASEMOVE_RS, bytes(res[0],encoding='utf-8'), bytes(res[1],encoding='utf-8'), b'')
                    elif len(res) == 3:
                        values = (DEF_ANALYSIS_BASEMOVE_RS, bytes(res[0],encoding='utf-8'), bytes(res[1],encoding='utf-8'), bytes(res[2],encoding='utf-8'))
                    s = struct.Struct('i10s10s10s')
                    if values is not None:
                        buff = s.pack(*values)
                        self.SendDate(socket, buff, len(buff))
            else:
                values = (DEF_ANALYSIS_FAIL_RQ,)
                s = struct.Struct('i')
                buff = s.pack(*values)
                self.SendDate(socket, buff, len(buff))

    def SendDate(self, sock, Pack, nLen):
        self.m_net.sendDate(sock, Pack, nLen)
