import signal
import time
import sys
import numpy as np
import pylab 



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

    
sstep = 0.02
loop = True
interaction = False


class NeuralLayer:
    def __init__(self,layerindex,myNeuralCount,activeFunc):
        self.index = layerindex
       # self.W=np.random.randn(myNeuralCount,lastNeuralCount)
        self.B=np.random.rand(myNeuralCount,1)
        self.neuralCount = myNeuralCount
        self.activeFunc = activeFunc
        self.lastNeural = None
        self.nextNeural = None
        
        
    def SetLastNeural(self,neural):
        self.lastNeural = neural
        self.W = np.random.randn(self.neuralCount,self.lastNeural.neuralCount)
        neural.nextNeural = self
        
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
        self.W = self.W - sstep * self.dw
        self.B = self.B - sstep * self.db
        if self.nextNeural!=None:
            self.nextNeural.ReviseWB()


            
            
def run_program():
    #pylab.plot(x , y )
    #pylab.show()
    
    n0 =   NeuralLayer(0,1,PureOutFunc)     
    
    n1 =   NeuralLayer(1,2,ReLU)
    n1.SetLastNeural(n0)
    
    n2 =   NeuralLayer(2,20,ReLU)
    n2.SetLastNeural(n1)
    
    n3 =   NeuralLayer(3,20,ReLU)
    n3.SetLastNeural(n2)
    
    n4 =   NeuralLayer(3,10,ReLU)
    n4.SetLastNeural(n3)
    
    n5 =   NeuralLayer(4,1,SigmoidFunc)
    n5.SetLastNeural(n4)
    
    last_nu = n5  

    
    Size = 100
    x=  np.linspace(0,10,Size)    
    y = SampleFunc(x)
    Y_H = 0

    X = x.reshape(1,Size)/ 100
    Y = y.reshape(1,Size) /100
    
    times = 0
    showindex = 0 
    global loop
    global interaction
    while loop:
        Y_H  = n0.Forward(X)
        J = costFunc( LossFunc(Y_H ,Y ) )
        showindex+=1
        if showindex==10000:
            showindex = 0
            times+=1
            print("cost value=",J,"@[",times,"*10000]")
            time.sleep(0.03)
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
                elif control=="continue":
                    interaction = False
                else:
                    interaction = False

        if J < 0.33:
            st = 0.02  
        if J < 0.32:
            break
        last_nu.Backward(Y)
        n1.ReviseWB()  
    

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