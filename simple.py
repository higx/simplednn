import signal
import time
import sys
import numpy as np
import pylab 
import yaml
import os

def SigmoidFunc(arrary):
    return 1/(1 + np.power( np.e ,-arrary) ) 

def PureOutFunc(x):
    return x


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def SampleFunc(x):
    return  np.abs( np.sin( x ) )* np.power(x,2)
    
def LossFunc(a,y):
    return  -(y * np.log(a)   + (1- y) * np.log( 1 - a) )

def costFunc( a ):
    return np.sum(a,axis=1,keepdims=True)/a.size

    
sstep = 0.008
loop = True
interaction = False


class NeuralLayer:
    def __init__(self,layerindex,myNeuralCount,activeFunc):
        self.index = layerindex
        self.B=np.random.rand(myNeuralCount,1)
        self.neuralCount = myNeuralCount
        self.activeFunc = activeFunc
        self.lastNeural = None
        self.nextNeural = None
        self.W = None
        
    def SetWB(self,W,B):
        self.W = W
        self.B = B
        
    def ShowShape(self):
        print("layer:",self.index)
        print("W",self.W.shape)
        print("B",self.B.shape)
        
    def SetLastNeural(self,neural):
        self.lastNeural = neural
        self.W = np.random.randn(self.neuralCount,self.lastNeural.neuralCount)
        neural.nextNeural = self
        self.ShowShape()
        
    def Forward(self,x):
        if self.lastNeural==None:
            out = self.activeFunc(x)
            self.X = x
            self.A = self.X
        else:
            self.Z = np.dot(self.W,x) + self.B
            self.A = self.activeFunc( self.Z )
            out = self.A
        
        if self.nextNeural!=None:
            return self.nextNeural.Forward(out)
        else:
            return out
        
    def Backward(self, Y_E ):
        if self.nextNeural==None:
            self.dz = self.A- Y_E
            self.dw = np.dot(self.dz,self.lastNeural.A.T)/Y_E.size
            self.db = np.sum(self.dz,axis=1,keepdims=True)/Y_E.size 
            self.lastNeural.Backward(Y_E)
        elif self.lastNeural!=None:
            self.dz = np.dot( self.nextNeural.W.T ,self.nextNeural.dz ) * dReLU(self.Z)
            self.dw = np.dot(self.dz,self.lastNeural.A.T)/Y_E.size
            self.db = np.sum(self.dz,axis=1,keepdims=True)/Y_E.size 
            self.lastNeural.Backward(Y_E)
            
        
    def ReviseWB(self ):
        if self.lastNeural==None:
            self.nextNeural.ReviseWB()
            return
        self.W = self.W - sstep * self.dw
        self.B = self.B - sstep * self.db
        if self.nextNeural!=None:
            self.nextNeural.ReviseWB()

    def OutputWB(self,config):
        thislayer = {}
        thislayer["index"] = self.index
        thislayer["W"] = DumpNumpyData(str(self.index)+"_W",self.W)
        thislayer["B"] = DumpNumpyData(str(self.index)+"_B",self.B)
        thislayer["neuralCount"] = self.neuralCount
        if self.activeFunc==PureOutFunc:
            thislayer["activeFunc"] = "PureOutFunc"
        elif self.activeFunc==ReLU:
            thislayer["activeFunc"] = "ReLU"
        elif self.activeFunc==SigmoidFunc:
            thislayer["activeFunc"] = "SigmoidFunc"
        else:
            thislayer["activeFunc"] = "None"
        config.append( thislayer )
        if self.nextNeural!=None:
            self.nextNeural.OutputWB(config)
        
            
def DumpNumpyData(filename,data):
    np.save("./data/"+filename, data)
    return filename
    
def LoadNumpyData(filename):
    return np.load("./data/"+filename+".npy")
    
            
def run_program():
    #pylab.plot(x , y )
    #pylab.show()
    Size = 100
    times = 0
    
    if os.path.exists("./wb.yaml"):
    
        with open('./wb.yaml', 'r') as yaml_file:
            yaml_obj = yaml.load(yaml_file.read())
            x = LoadNumpyData(yaml_obj["input"])
            y = LoadNumpyData(yaml_obj["output"])
            lastLayer = None
            for layer in yaml_obj["layerconfig"]:
                func = layer["activeFunc"]
                if func=="PureOutFunc":
                    print("create PureOutFunc")
                    neuralLay = NeuralLayer(layer["index"],layer["neuralCount"],PureOutFunc) 
                    n0 = neuralLay
                elif func=="ReLU":
                    print("create ReLU")
                    neuralLay = NeuralLayer(layer["index"],layer["neuralCount"],ReLU) 
                elif func=="SigmoidFunc":
                    print("create SigmoidFunc")
                    neuralLay = NeuralLayer(layer["index"],layer["neuralCount"],SigmoidFunc)  
                if lastLayer!=None:
                    neuralLay.SetLastNeural(lastLayer)
                neuralLay.SetWB(LoadNumpyData(layer["W"]),LoadNumpyData(layer["B"]))
                lastLayer = neuralLay
                last_nu = neuralLay 
    else:
        n0 =   NeuralLayer(0,1,PureOutFunc)     
        
        n1 =   NeuralLayer(1,2,ReLU)
        n1.SetLastNeural(n0)
        
        n2 =   NeuralLayer(2,20,ReLU)
        n2.SetLastNeural(n1)
        
        n3 =   NeuralLayer(3,20,ReLU)
        n3.SetLastNeural(n2)
        
        n4 =   NeuralLayer(4,10,ReLU)
        n4.SetLastNeural(n3)
        
        n5 =   NeuralLayer(5,1,SigmoidFunc)
        n5.SetLastNeural(n4)
        
        last_nu = n5  
  
        x=  np.linspace(0,10,Size)    
        y = SampleFunc(x)

    

    Y_H = 0

    X = x.reshape(1,Size)/ 100
    Y = y.reshape(1,Size) /100
    
    
  
    showindex = 0 
    global loop
    global interaction
    global sstep
    while loop:
        Y_H  = n0.Forward(X)
        J = costFunc( LossFunc(Y_H ,Y ) )
        showindex+=1
        if showindex==10000:
            showindex = 0
            times+=1
            print("cost value=",J,"@[",times,"*10000]")
            #time.sleep(0.03)
            if interaction:
                control = input("input:")
                if control=="exit":
                    loop = False
                elif control=="showorgpic":
                    pylab.plot(x , y )
                    pylab.show()
                elif control=="showpic":
                    show_y = Y_H * 100
                    pylab.plot(x , show_y[0] )
                    pylab.show()
                elif control=="dump":
                    dumpinfo = {}
                    dumpinfo["input"] = DumpNumpyData("X",x)
                    dumpinfo["output"] = DumpNumpyData("Y",y)
                    config=[]
                    n0.OutputWB(config)
                    dumpinfo["layerconfig"] = config
                    with open("./wb.yaml", "w") as yaml_file:
                       yaml.dump(dumpinfo, yaml_file,default_flow_style=False)
                elif len(control)>2 and (control[0]=="+" or control[0]=="-"):
                    f = float(control)  
                    sstep += f
                    print( sstep )
                elif control=="continue":
                    interaction = False
                else:
                    interaction = False

        if J < 0.33:
            sstep = 0.02  
        if J < 0.32:
            break
        last_nu.Backward(Y)
        n0.ReviseWB()  
    

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)
    global interaction
    try:
        print("=========Human interaction===========")    
        interaction = True
    except RuntimeError:
        interaction = True


    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_gracefully)

if __name__ == '__main__':
    # store the original SIGINT handler

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    run_program()