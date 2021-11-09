from Kernel import *
import time
import threading
from command_analysis import *

if __name__ == '__main__':
    kernel = Kernel()
    kernel.initNet()
    while True:
        print("server running...")
        print("pthread count",threading.active_count())
        time.sleep(30)