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

       
Size = 100 
sstep = 0.01
loop = True
interaction = False
beta1  = 0.9
beta2  = 0.99
#sigma = 0.4
#mu = 0

class NeuralLayer:
    def __init__(self,layerindex,myNeuralCount,activeFunc):
        self.index = layerindex
        self.W = None
        self.B=np.random.rand(myNeuralCount,1)
        self.neuralCount = myNeuralCount
        self.activeFunc = activeFunc
        self.lastNeural = None
        self.nextNeural = None

        self.dw =None
        self.db = None
        self.vdw = 0
        self.vdb = 0
        self.sdw = 0
        self.sdb = 0
        self.SimulateW = None
        self.SimulateB = None

        
    def SetWB(self,W,B):
        self.W = W
        self.B = B

    def SetVWB(self, W, B):
        self.vdw = W
        self.vdb = B

    def SetSWB(self,W,B):
        self.sdw = W
        self.sdb = B


    def SetSimulateWB(self, start ,all_concatenate):
        self.SimulateW = all_concatenate[start:start+self.W.size].reshape(self.W.shape)
        self.SimulateB = all_concatenate[start + self.W.size:
                                        start + self.W.size+self.B.size].reshape(self.B.shape)
        assert((self.SimulateW==self.W).all())
        assert((self.SimulateB == self.B).all())
        return start+self.SimulateW.size + self.SimulateB.size

        
    def ShowShape(self):
        print("layer:",self.index," W:",self.W.shape," B:",self.B.shape)
        
    def SetLastNeural(self,neural):
        self.lastNeural = neural
        self.W = np.random.randn(self.neuralCount,self.lastNeural.neuralCount)* np.sqrt(2/self.lastNeural.neuralCount)
        neural.nextNeural = self
        self.ShowShape()
        
    # output with SimulateWB
    def SimulateOutput(self,x):
        if self.lastNeural==None:
            out = self.activeFunc(x)
        else:
            Z = np.dot(self.SimulateW, x) + self.SimulateB
            A = self.activeFunc( Z )
            out = A
        if self.nextNeural!=None:
            return self.nextNeural.SimulateOutput(out)
        else:
            return out
        
    #just output 
    def Output(self,x):
        if self.lastNeural==None:
            out = self.activeFunc(x)
        else:
            Z = np.dot(self.W,x) + self.B
            A = self.activeFunc( Z )
            out = A
        if self.nextNeural!=None:
            return self.nextNeural.Output(out)
        else:
            return out

    #save template A for backward
    def Forward(self, x):
        if self.lastNeural == None:
            out = self.activeFunc(x)
            self.X = x
            self.A = self.X
        else:
            self.Z = np.dot(self.W, x) + self.B
            self.A = self.activeFunc(self.Z)
            out = self.A

        if self.nextNeural != None:
            return self.nextNeural.Forward(out)
        else:
            return out

    def get_W_BConcatenate(self):
        wrs = self.W.reshape(-1)
        brs = self.B.reshape(-1)
        return np.concatenate((wrs, brs), axis=0)

    def get_dw_dbConcatenate(self):
        dwrs = self.dw.reshape(-1)
        dbrs = self.db.reshape(-1)  
        return np.concatenate((dwrs,dbrs), axis=0) 

    def get_WB_Info(self):
        wshape = self.W.shape
        wlength  = len( self.W.reshape(-1) )
        bshape = self.B.shape
        blength  = len( self.B.reshape(-1) )
        
        self.WRange = (self.lastNeural.BRange[1],self.lastNeural.BRange[1] +wlength)
        
        self.BRange = (self.WRange[1],self.WRange[1] +blength) 
        
        WBInfo = {"layer":self.index,"wlen":wlength,"blen":blength,"wrange":self.WRange,"brange":self.BRange}
        return WBInfo
        
             
        
    def GradCheck(self,X,Y):
    
        w_b_array = []
        theta_approx = []
        theta = []
        Epsilon = 10**-7
        nextNeural = self.nextNeural

        while nextNeural !=None:
            w_b_array = np.concatenate((w_b_array,nextNeural.get_W_BConcatenate()), axis=0)
            theta = np.concatenate(
                (theta, nextNeural.get_dw_dbConcatenate()), axis=0)
            nextNeural = nextNeural.nextNeural

        #set simmulate w_b to neural
        nextNeural = self.nextNeural
        setIndex = 0
        while nextNeural != None:
            #print("start:", setIndex, " set layer simulate", nextNeural.index)
            setIndex = nextNeural.SetSimulateWB(setIndex, w_b_array)
            nextNeural = nextNeural.nextNeural

        assert(theta.size == w_b_array.size)


        for i in range(len(w_b_array)):
            org_value = w_b_array[i]

            w_b_array[i] = org_value + Epsilon
            y = self.SimulateOutput(X)
            J_add = costFunc(LossFunc(y, Y))

            #org_y = self.Output(X)
            #assert( (Yorg_y==y).all() )
        
            w_b_array[i] = org_value - Epsilon
            y = self.SimulateOutput(X)
            J_minus = costFunc(LossFunc(y, Y))

            dtheta = (J_add - J_minus) / (2 * Epsilon)
            theta_approx.append(dtheta[0][0])
            w_b_array[i] = org_value

        theta_approx = np.array(theta_approx)
        assert( theta_approx.size == theta.size )
        #print(theta_approx)
        #print(theta)

        check = np.linalg.norm(theta_approx - theta) / \
            (np.linalg.norm(theta_approx)+np.linalg.norm(theta))

        print( check )
        if check < 10**-7:
            print("grade check: great" )
        elif 10**-7 < check and check < 10**-5:
            print("grade check: normal")
        elif 10**-3 < check:
            print("grade check: worse")

        
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

        #self.W = self.W - sstep * self.dw
        #self.B = self.B - sstep * self.db
        self.vdw = beta1 * self.vdw + (1 - beta1) * self.dw 
        self.vdb = beta1 * self.vdb + (1 - beta1) * self.db
        self.sdw = beta2 * self.sdw + (1 - beta2) * self.dw * self.dw
        self.sdb = beta2 * self.sdb + (1 - beta2) * self.db * self.db

        self.W = self.W - sstep * ( self.vdw / np.sqrt(self.sdw + 10**-8))
        self.B = self.B - sstep * (self.vdb / np.sqrt(self.sdb + 10**-8))
        if self.nextNeural!=None:
            self.nextNeural.ReviseWB()

    def OutputWB(self,config):
        folder = os.path.exists("./data")
        if not folder:
            os.makedirs("./data")
        thislayer = {}
        thislayer["index"] = self.index
        thislayer["W"] = DumpNumpyData(str(self.index)+"_W",self.W)
        thislayer["B"] = DumpNumpyData(str(self.index)+"_B",self.B)
        thislayer["vdw"] = DumpNumpyData(str(self.index)+"_vdw", self.vdw)
        thislayer["vdb"] = DumpNumpyData(str(self.index)+"_vdb", self.vdb)
        thislayer["sdw"] = DumpNumpyData(str(self.index)+"_sdw", self.sdw)
        thislayer["sdb"] = DumpNumpyData(str(self.index)+"_sdb", self.sdb)
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
    global loop
    global interaction
    global sstep

    times = 0
    
    if os.path.exists("./wb.yaml"):
    
        with open('./wb.yaml', 'r') as yaml_file:
            yaml_obj = yaml.load(yaml_file.read())
            x = LoadNumpyData(yaml_obj["input"])
            y = LoadNumpyData(yaml_obj["output"])
            sstep = yaml_obj["sstep"]
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
                neuralLay.SetVWB(LoadNumpyData(layer["vdw"]), LoadNumpyData(layer["vdb"]))
                neuralLay.SetSWB(LoadNumpyData(layer["sdw"]), LoadNumpyData(layer["sdb"]))
                lastLayer = neuralLay
                last_nu = neuralLay 
    else:
        n0 =   NeuralLayer(0,1,PureOutFunc)     
        
        n1 =   NeuralLayer(1,10,ReLU)
        n1.SetLastNeural(n0)
        
        n2 =   NeuralLayer(2,20,ReLU)
        n2.SetLastNeural(n1)
          
        n3 =   NeuralLayer(3,10,ReLU)
        n3.SetLastNeural(n2)

        n4 =   NeuralLayer(4,1,SigmoidFunc)
        n4.SetLastNeural(n3)
        
        last_nu = n4 
  
        x=  np.linspace(0,10,Size)    
        y = SampleFunc(x)

    

    Y_H = 0

    X = x.reshape(1,Size)/ Size
    Y = y.reshape(1,Size) /Size
    
    
    gradcheck = False
    showindex = 0 

    while loop:
        Y_H  = n0.Forward(X)
        J = costFunc( LossFunc(Y_H ,Y ) )  
        last_nu.Backward(Y)
      
        showindex+=1
        if showindex==10000:
            showindex = 0
            times+=1
            print("cost value=",J,"@[",times,"*10000]")
            if gradcheck:
                n0.GradCheck(X, Y)
            #time.sleep(0.03)
            if J < 0.3:
                break
            if interaction:
                control = input("input:")
                if control=="exit":
                    loop = False
                elif control=="showpic":
                    show_y = Y_H * 100
                    pylab.plot(x , show_y[0] )
                    pylab.plot(x , y )
                    pylab.show()
                elif control=="showtest":
                    test_size = 100
                    test_x=  np.linspace(0,20,test_size)  
                    test_y = SampleFunc(test_x)
                    test_X = test_x.reshape(1,test_size)/ test_size
                    test_Y = n0.Output(test_X)
                    show_test_y = test_Y * test_size
                    pylab.plot(test_x , show_test_y[0] )
                    pylab.plot(test_x , test_y )
                    pylab.show()
                elif control=="dump":
                    dumpinfo = {}
                    dumpinfo["sstep"] = sstep
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
                print( "sstep:", sstep)

            n0.ReviseWB()

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)
    global interaction
    try:
        print("=========Human interaction=========")    
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
