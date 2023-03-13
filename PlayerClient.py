from enum import Enum

class State(Enum):
    ReadyToDisplay = 1
    Playing = 2
    Finish = 3

class InfoProcessor:
    def __int__(self, writepath):
        self.idx = 0
        self.writepath = writepath
        self.msgs = []

    def initialmsg(self):
        self.currentwrite = self.writepath + "target%d.zip"%self.idx
        self.msgs = []

    def derivateState(self,pic):
        pass  ### esay job

    def deriveinfo(self, pic):
        return False #### check if it's valid, append the message, need to borrow from previous parser

    def savemessage(self):
        f = open(self.writepath,"wb")
        for pg in self.msgs:
            f.write(pg)
        f.close()
    def cleanandswitchtonextfile(self):
        self.idx = self.idx + 1
         ## about unzip, we can do it later, unzip, extract the target name, etc, or data at first few characters are the name

class SignalInput(Enum):
    Empty = 0
    Ready = 1 ### State 1 to 2
    Pause = 2 ### State 2 to 2
    ### will see whether we need Previous
    Next  = 3 ### State 2 to 2
    SaveComplete = 4 ### State 3 to 1

def sendkey(s):
    pass

import time
class NuclearPowerReceiver:

    def __int__(self,infop,camera,sleeptime):
        self.infop = infop
        self.camera = camera
        self.sleeptime = sleeptime

    def getpic(self):
        self.camera.getpic()

    def run(self):
        self.infop.initialmsg()

        while True:
            state = self.infop.derivateState()
            if state == State.ReadyToDisplay:
                self.SendSignal(SignalInput.Ready)
                time.sleep(self.sleeptime)

            elif state == State.Playing:
                rtn = self.infop.deriveinfo()
                if rtn:
                    self.SendSignal(SignalInput.Next)
                else:
                    print("broken info pic, wait for a bit")
                    time.sleep(self.sleeptime)
                    self.SendSignal(SignalInput.Pause)
            elif state == State.Finish:
                if len(self.msgs) > 0:
                    self.infop.savemessage()
                    self.infop.cleanandswitchtonextfile()
                    self.SendSignal(SignalInput.SaveComplete)
                else:
                    time.sleep(self.sleeptime)
                    self.SendSignal(SignalInput.SaveComplete)


    def SendSignal(self,sig):
        if sig == SignalInput.Empty:
            sendkey("e")
        elif sig == SignalInput.Ready:
            sendkey("r")
        elif sig == SignalInput.Next:
            sendkey("n")
        elif sig == SignalInput.Pause:
            sendkey("p")
        elif sig == SignalInput.SaveComplete:
            sendkey("s")
