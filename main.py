#scan for nxt devices
# def bluetooth_low_energy_scan(timeout=10):
#     svc = DiscoveryService()
#     devs = svc.discover(timeout)

#     print('found',len(devs),'Bluetooth Low Energy (Smart) devices:')

#     if devs:
#         for u,n in devs.items():
#             print(u,n)

#     return devs

import bluetooth as bt
import socket, pickle

def send(msg, s):
    buf = bytearray(b'\x00\x09\x02\xFF' + bytes(msg))
    print("Sent message " + msg)
    s.send(buf)    

if __name__ == '__main__':
    mac = '00:16:53:0D:C9:E7'
    serverMACAddress = mac
    port = 1
    s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    s.connect((serverMACAddress,port))
    send('test', s)
    s.close()