### Think about the interfact flow
import cv2
import tkinter as tk
import pynput
from pynput import keyboard
from threading import Thread,Lock
import cv2

## Let's make this sample

RUNNING = "prod"

## Just Stop at each picture
class PicPlayer:

    def __init__(self):
        self.displaystate = False
        self.displayloc = Lock()
        self.imgs = []
        self.processed = 0

    def displayimg(self,img):
        self.displayloc.acquire()
        self.displaystate = True
        imgcv2 = cv2.imread(img)
        while True:
            cv2.imshow("cool",imgcv2)
            cv2.waitKey(500)
            if not self.displaystate:
                cv2.destroyAllWindows()
                break
        self.displayloc.release()
        ### one display one time

    def displayready(self):
        self.displaystate = False
        x = "C:\\Users\\zheshi\\PycharmProjects\\ControlledPlayer\\source\\ready.bmp"
        tr = Thread(target=self.displayimg,args=[x])
        tr.start()

    def displaynext(self):
        if self.iscurrentcomplete():
            self.displaycomplete()
        else:
            self.displaystate = False
            tr = Thread(target=self.displayimg, args=[self.imgs[self.processed]])
            tr.start()
            self.processed = self.processed + 1


    def displaycomplete(self):
        self.displaystate = False
        x = r"C:\\Users\zheshi\\PycharmProjects\\ControlledPlayer\\source\\complete.bmp"
        tr = Thread(target=self.displayimg,
                    args=[x])
        tr.start()

    ### uploading time, well display sth, once finishi, display read
    def upload(self, nextvideo):
        self.processed = 0
        self.imgs = [r"C:\Users\zheshi\PycharmProjects\ControlledPlayer\source\data1.jpg",r"C:\Users\zheshi\PycharmProjects\ControlledPlayer\source\data2.jpg"]

    def iscurrentcomplete(self):
        if self.processed >= len(self.imgs):
            return  True
        else:
            return False

from enum import Enum

class SignalInput(Enum):
    Empty = 0
    Ready = 1 ### State 1 to 2
    Pause = 2 ### State 2 to 2
    ### will see whether we need Previous
    Next  = 3 ### State 2 to 2
    SaveComplete = 4 ### State 3 to 1

class State(Enum):
    CachingData = 0 ## This should be invisible to clients
    ReadyToDisplay = 1
    Playing = 2
    Finish = 3

import os

class NuclearReactor:

    def __init__(self,path,picplay):
        self.avipath = path
        self.processed = []
        self.player = picplay
        if self.checkneedtocache():
            self.preparezipinbackgroud()
        self.state = State.ReadyToDisplay

    def checkneedtocache(self):
        if len(os.listdir(self.avipath)) > len(self.processed):
            return False
        else:
            return True
    ### zips, download zips, translate to pictures and AVIs, copy AVIs to destination, open AVIs ~~~ so need to constantly copy AVI ~~~ amybe access from sever
    def preparezipinbackgroud(self):
            pass
    def uploadforplayer(self):
            self.player.upload("somefile")

    def handlesignal(self, input):
        if input == SignalInput.Empty:
            pass
        elif input == SignalInput.Ready:
            if self.state == State.ReadyToDisplay:
                self.state = State.Playing
                self.player.displaynext()
            else:
                print("wrong signal, doing nothing")

        elif input == SignalInput.Next:
            if self.state == State.Playing:
                if self.player.iscurrentcomplete():
                    self.state = State.Finish
                    self.player.displaycomplete()
                else:
                    self.player.displaynext()
            else:
                print("wrong signal, doing nothing")

        elif input == SignalInput.Pause:
            if self.state == State.Playing:
                print("waiting for image recover")
        elif input == SignalInput.SaveComplete:
            if self.state == State.Finish:
                if self.checkneedtocache():
                    self.state = State.CachingData
                    self.preparezipinbackgroud()
                    self.state = State.ReadyToDisplay
                    self.uploadforplayer()
                    self.player.displayready()
                else:
                    self.state = State.ReadyToDisplay
                    self.uploadforplayer()
                    self.player.displayready()
            else:
                print("wrong signal, doing nothing")
        else:
            pass
    ### key/signal translation
    def getSignalInput(self,key):
        if key.char == "r":
            return SignalInput.Ready
        elif key.char == "n":
            return SignalInput.Next
        elif key.char == "p":
            return SignalInput.Pause
        elif key.char == "s":
            return SignalInput.SaveComplete
        else:
            return SignalInput.Empty

    def onkey(self,key):
        pass
    def releasekey(self,key):
        signal = self.getSignalInput(key)
        self.handlesignal(signal)

    def run(self):
        self.uploadforplayer()
        self.player.displayready()
        while True:
            with keyboard.Listener(
                    on_press=lambda key: self.onkey(key),
                    on_release=lambda key: self.releasekey(key)) as listener:
                listener.join()


if __name__ == "__main__":
    RUNNING = "trail"
    p = PicPlayer()
    r = NuclearReactor("./",p)
    r.run()
