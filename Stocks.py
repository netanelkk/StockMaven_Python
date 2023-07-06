# python3 .\main.py 127.0.0.1:8081 127.0.0.1:8080 127.0.0.1:8082
# python3 .\main.py 127.0.0.1:8080 127.0.0.1:8081 127.0.0.1:8082
# python3 .\main.py 127.0.0.1:8082 127.0.0.1:8081 127.0.0.1:8080

#.\main.py 127.0.0.1:8081 127.0.0.1:8080 127.0.0.1:8082
#.\main.py 127.0.0.1:8080 127.0.0.1:8081 127.0.0.1:8082
#.\main.py 127.0.0.1:8082 127.0.0.1:8081 127.0.0.1:8080
import socket
import sys
import threading
import traceback
from datetime import datetime
import json
from time import sleep

HEADERSIZE = 10


class State(object):
    def __init__(self, args):
        self.host = self._parseHost(args[0])
        self.siblings = [self._parseHost(x) for x in args[1:]]
        self.connections = {}
        self.dicti = {"ts": datetime.timestamp(datetime.now())}
        ts = datetime.timestamp(datetime.now())
        self.dicti = {"empty_flag" : ["created at",ts]}



    def siblingFile(self, sibling):
        (con, f) = self.connections.get(sibling, (None, None))
        try:
            if con is None or con.fileno() == -1:
                con = socket.create_connection(sibling)
                f = con.makefile(mode='rw', encoding='utf-8')
                f.write('sibling\n')
                self.connections[sibling] = (con, f)
        except:
            sleep(1)

        return f

    def _parseHost(self, hostAndPort):
        parts = hostAndPort.split(':')
        host = parts[0]
        port = int(parts[1])
        return (host, port)

    def client(self, cmd, f):
        ts = datetime.timestamp(datetime.now())
        print('Got command: ', cmd)
        print("dicti type")
        print(type(self.dicti))


        if (cmd[1] == "my_dictionary"):
            print("dict if")
            if cmd[1].get("ts")>self.dicti.get("ts"):
                jsonDict = json.dumps(cmd[1])
                self.dicti = json.loads(jsonDict)
                print("need to change the dicti")


        for sibling in self.siblings:
            con = self.siblingFile(sibling)

            # data = pickle.dumps([cmd,ts])
            # data = bytes(f"{len(data):<{HEADERSIZE}}", 'utf-8')+data
            if self.dicti.get("empty_flag") is not None:
                jsonString = json.dumps([cmd, "empty_flag" ,ts])
            else:
                jsonString = json.dumps([cmd,ts])

            if cmd[1] == "empty_flag":
                print("need my dici")
                if self.dicti.get("empty_flag") is not None:
                    f.write("the socket is empty too")
                else:
                    jsonString = json.dumps([cmd, "my_dictionary", ts, self.dicti])
                    print("send my dict")

            try:
                con.write(jsonString)
                con.write('\n')
                con.flush()
            except:
                f.write("one of the sibling is not valid")

        rev = cmd[::-1]
        f.write(rev)
        f.write("\n")
        x = cmd.split()

        if (cmd.count(' ')==2):
            if(x[0]=="set"):
                if self.dicti.get("empty_flag") is not None:
                    self.dicti.pop("empty_flag")
                    print("remove emptyflag")
                if self.dicti.get(x[1]) is not None:
                    if (self.dicti.get(x[1])[1] < ts):
                        self.dicti.update({x[1]: [x[2], ts]})
                        self.dicti.update({"ts": ts})

                    else:
                        f.write("The value cannot register\n")
                else:
                    self.dicti.update({x[1]: [x[2], ts]})
                    self.dicti.update({"ts": ts})

        if (cmd.count(' ') == 1):
            x = cmd.split()
            if (x[0] == "get"):
                if self.dicti.get(x[1]) is not None:
                    f.write("\n" + self.dicti.get(x[1])[0])
                else:
                    f.write("The value " + x[1] + " is not in the dictionary please set it\n")
        f.flush()

    def sibling(self, cmd, f):
        print("From Sibling: ", cmd)
        f.write('Processed\n')
        # f.wrimdte()

        # fullmsg = b''
        # fullmsg += cmd
        # msg = pickle.loads(fullmsg[HEADERSIZE:])
        # x = msg[0].split()

        cmd = json.loads(cmd)
        print(type(cmd))

        if isinstance(cmd[1], dict):
            if cmd[1].get("ts")>self.dicti.get("ts"):
                jsonDict = json.dumps(cmd[1])
                self.dicti = json.loads(jsonDict)
                print("need to change the dicti")

        # need to check
        print(cmd)
        print("\ncmd[0] : ")
        print(cmd[0])
        print(cmd[0].count(' '))
        if (cmd[0].count(' ')==2):
            x = cmd[0].split()
            print(x[0])
            print(x[1])
            print(x[2])
            if(x[0]=="set"):
                if self.dicti.get("empty_flag") is not None:
                    self.dicti.pop("empty_flag")
                    print("remove emptyflag")
                if self.dicti.get(x[1]) is not None:
                    if (self.dicti.get(x[1])[1] < x[1]):
                        self.dicti.update({x[1]: [x[2], cmd[2]]})
                        self.dicti.update({"ts": cmd[2]})
                    else:
                        f.write("The value cannot register\n")
                else:
                    self.dicti.update({x[1]: [x[2], cmd[2]]})
                    self.dicti.update({"ts": cmd[2]})

        #need to check

        if (cmd[0].count(' ') == 1):
            x = cmd[0].split()
            if (x[0] == "get"):
                if self.dicti.get(x[1]) is not None:
                    f.write("\n" + self.dicti.get(x[1])[0])
                else:
                    f.write("The value " + x[1] + " is not in the dictionary please set it\n")

        f.flush()

    def run(self, s, addr):
        try:
            with s, s.makefile(mode='rw', encoding='utf-8') as f:
                firstLine = f.readline()
                isSibling = firstLine.rstrip('\n\r \t') == 'sibling'
                print('first line: ', firstLine, isSibling)

                if not isSibling:
                    self.client(firstLine, f)
                else:
                    print("Got sibling connection from ", addr)

                for line in f:
                    if isSibling:
                        self.sibling(line, f)
                    else:
                        self.client(line, f)
        except:
            print('Connection failed with ', addr)
            traceback.print_exc()


state = State(sys.argv[1:])
listener = socket.socket()
listener.bind(state.host)
listener.settimeout(0.2)  # don't hang
listener.listen(2)

print("Ready! Listening on ", state.host, "siblings: ", state.siblings)

while True:
    try:
        clientSocket, addr = listener.accept()
        print("Accepted request from: ", addr)
        threading.Thread(target=state.run, args=[clientSocket, addr]).start()
        print(socket.timeout)
    except socket.timeout:
        pass