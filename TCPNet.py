from socket import *
from threading import Thread
from Packdef import *

def recv_ptr(client_socket,self):
    global quitFlag
    while quitFlag:
        # 先发包大小，再发包内容
        recv_date = client_socket.recv(4)
        s_size = struct.Struct('<i')
        size_date = s_size.unpack(recv_date)
        print(size_date)

        recv_date = client_socket.recv(size_date[0])
        s_size = struct.Struct('i100s')
        size_date = s_size.unpack(recv_date)

        if size_date[0] == DEF_ANALYSIS_QUIT_RQ:
            self.sockets.remove(client_socket)
            client_socket.close()
            print("exit client",client_socket)
            break

        self.Kernel.DealDate(client_socket,size_date,size_date)
        # print(size_date)
        # if size_date[0] == DEF_ANALYSIS_CHATTER_RQ:
        #     values = (DEF_ANALYSIS_CHATTER_RS, b'command_analysis server state normal')
        #     s = struct.Struct('i100s')
        #
        #     buff = s.pack(*values)
        #
        #     # buff = ctypes.create_string_buffer(s.size)
        #     # pack_date = s.pack_into(buff, s.size, *values)
        #     size_date = len(buff)
        #
        #     value_size = (size_date,)
        #     s_size = struct.Struct('i')
        #     buff_size = s_size.pack(*value_size)
        #
        #     res = client_socket.send(buff_size)
        #
        #     client_socket.send(buff)


def accept_ptr(server_socket,self):
    global quitFlag
    while quitFlag:
        client_socket, client_info = server_socket.accept()
        if client_socket:
            print('client connect', client_info)
            self.sockets.append(client_socket)
            tid = Thread(target=recv_ptr, args=(client_socket,self))
            tid.start()
            self.pthreads.append(tid)


class TCPNet():
    def __init__(self,kernel):
        self.m_sockListen= 0
        self.quitFlag = True
        self.sockets = []
        self.pthreads = []
        self.Kernel = kernel

    def InitNetWork(self):
        # 创建serversocket
        self.m_sockListen = socket(AF_INET, SOCK_STREAM)
        # 绑定
        self.m_sockListen.bind((service_IP, service_Port))
        # 监听
        self.m_sockListen.listen()

        t = Thread(target=accept_ptr, args=(self.m_sockListen,self))
        t.start()
        return True

    def sendDate(self,sock,date,nLen):
        # 先发包大小
        val_size = (nLen,)
        s_size = struct.Struct('i')
        buff_size = s_size.pack(*val_size)

        res = sock.send(buff_size)
        if res == 0:
            return False
        # 再发包内容
        res = sock.send(date)
        if res == 0:
            return  False;
        return True