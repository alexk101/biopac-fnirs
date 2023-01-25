# client.py

import socket
import struct
from typing import Callable
import threading
from pathlib import Path

mark: Callable[[int], bytes] = lambda x: x.to_bytes(8, 'little')
HOST = "129.79.192.162"  # The server's hostname or IP address
PORT = 6350  # The port used by the server

MARKERS = {
    'trail_start' : mark(1),
    'trail_end' : mark(2),
    'baseline_start' : mark(251),
    'baseline_end' : mark(252),
}

VALID_OPTS = list(range(1,len(list(MARKERS.keys()))+1))

INSTR = '\n'.join([f"({num}) {marker}" for num, marker in zip(VALID_OPTS, MARKERS.keys())])



# In bytes
STATUS_LEN = 2
TIME_LEN = 4

QUEUE = []

class Client:
    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.output = Path('./output.txt').resolve()
    
    def run(self):
        with self.socket as s:
            with open(self.output, 'w') as fp:
                s.connect((HOST, PORT))
                while 1:
                    arr1 = s.recv(STATUS_LEN)
                    # print(f"arr1 {arr1!r}")
                    if arr1[0] == 38 and arr1[1] == 108:
                        arr2 = s.recv(436)
                        # print(f"arr2 {arr2[-4:]!r}")
                        val = struct.unpack('f', arr2[-TIME_LEN:])[0]
                        fp.write(f'{val}\n')
                    else:
                        fp.write(f"Connection broken!\n")
                    if len(QUEUE):
                        val = QUEUE.pop(0)
                        if val == 'q':
                            break
                        s.send(MARKERS[list(MARKERS.keys())[val-1]])

def get_marker():
    while 1:
        print(INSTR)
        val = input()
        if val == 'q':
            QUEUE.append(val)
            break
        if val.isdigit() and int(val) not in VALID_OPTS:
            print(f'Invalid option')
        else:
            print(f'sending {int(val)}')
            QUEUE.append(int(val))


if __name__ =="__main__":
    test = Client()
    t1 = threading.Thread(target=get_marker)
    t2 = threading.Thread(target=test.run)
 
    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
 
    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()