import signal
import time
import sys
import numpy as np
import pylab
import yaml
import os

beta1 = 0.9
beta2 = 0.99

interaction = False


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



def DumpNumpyData(filename, data):
    np.save("./data/"+filename, data)
    return filename


def LoadNumpyData(filename):
    return np.load("./data/"+filename+".npy")


def SigmoidFunc(arrary):
    return 1/(1 + np.power(np.e, -arrary))


def Normalize(input, gamma, beta):
    Epsilon = 10**-7
    mu = np.sum(input) / input.size
    sigma2 = np.sum(np.power((input-mu), 2)) / input.size
    z_norm = (input - mu) / np.power(sigma2 + Epsilon, 0.5)
    return z_norm * gamma + beta


def PureOutFunc(x):
#    return Normalize(x, 1, 0.5)
    return x

def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def SampleFunc(x):
    return np.abs(np.sin(x)) * np.power(x, 2)


def LossFunc(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def costFunc(a):
    return np.sum(a, axis=1, keepdims=True)/a.size


class NeuralLayer:
    def __init__(self, myNeuralCount, activeFunc):
        self.index = 0
        self.W = None
        self.B = np.random.rand(myNeuralCount, 1)
        self.neuralCount = myNeuralCount
        self.activeFunc = activeFunc
        self.lastNeural = None
        self.nextNeural = None

        self.dw = None
        self.db = None
        self.vdw = 0
        self.vdb = 0
        self.sdw = 0
        self.sdb = 0
        self.SimulateW = None
        self.SimulateB = None
    
    def NextLayer(self,  myNeuralCount, activeFunc):
        nextlayer = NeuralLayer( myNeuralCount, activeFunc)
        nextlayer.index = self.index + 1
        nextlayer.SetLastNeural(self)
        return nextlayer

    def Head(self):
        lastNeural = self
        while lastNeural.lastNeural != None:
            lastNeural = lastNeural.lastNeural
        return lastNeural

    def End(self):
        nextNeural = self
        while nextNeural.nextNeural != None:
            nextNeural = nextNeural.nextNeural
        return nextNeural

    def SetWB(self, W, B):
        self.W = W
        self.B = B

    def SetVWB(self, W, B):
        self.vdw = W
        self.vdb = B

    def SetSWB(self, W, B):
        self.sdw = W
        self.sdb = B

    def SetSimulateWB(self, start, all_concatenate):
        self.SimulateW = all_concatenate[start:start +
                                         self.W.size].reshape(self.W.shape)
        self.SimulateB = all_concatenate[start + self.W.size:
                                         start + self.W.size+self.B.size].reshape(self.B.shape)
        assert((self.SimulateW == self.W).all())
        assert((self.SimulateB == self.B).all())
        return start+self.SimulateW.size + self.SimulateB.size

    def ShowShape(self):
        print("layer:", self.index, " W:", self.W.shape, " B:", self.B.shape)

    def SetLastNeural(self, neural):
        self.lastNeural = neural
        self.W = np.random.randn(
            self.neuralCount, self.lastNeural.neuralCount) * np.sqrt(2/self.lastNeural.neuralCount)
        neural.nextNeural = self
        self.ShowShape()

    # output with SimulateWB
    def SimulateOutput(self, x):
        if self.lastNeural == None:
            out = self.activeFunc(x)
        else:
            Z = np.dot(self.SimulateW, x) + self.SimulateB
            A = self.activeFunc(Z)
            out = A
        if self.nextNeural != None:
            return self.nextNeural.SimulateOutput(out)
        else:
            return out

    #just output
    def Output(self, x):
        if self.lastNeural == None:
            out = self.activeFunc(x)
        else:
            Z = np.dot(self.W, x) + self.B
            A = self.activeFunc(Z)
            out = A
        if self.nextNeural != None:
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
        return np.concatenate((dwrs, dbrs), axis=0)

    def get_WB_Info(self):
        wshape = self.W.shape
        wlength = len(self.W.reshape(-1))
        bshape = self.B.shape
        blength = len(self.B.reshape(-1))

        self.WRange = (
            self.lastNeural.BRange[1], self.lastNeural.BRange[1] + wlength)

        self.BRange = (self.WRange[1], self.WRange[1] + blength)

        WBInfo = {"layer": self.index, "wlen": wlength,
                  "blen": blength, "wrange": self.WRange, "brange": self.BRange}
        return WBInfo

    def GradCheck(self, X, Y):

        w_b_array = []
        theta_approx = []
        theta = []
        Epsilon = 10**-7
        nextNeural = self.nextNeural

        while nextNeural != None:
            w_b_array = np.concatenate(
                (w_b_array, nextNeural.get_W_BConcatenate()), axis=0)
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
        assert(theta_approx.size == theta.size)
        #print(theta_approx)
        #print(theta)

        check = np.linalg.norm(theta_approx - theta) / \
            (np.linalg.norm(theta_approx)+np.linalg.norm(theta))

        print(check)
        if check < 10**-7:
            print("grade check: great")
        elif 10**-7 < check and check < 10**-5:
            print("grade check: normal")
        elif 10**-3 < check:
            print("grade check: worse")
        else:
            print("grade check: worst")

    def Backward(self, Y_E):
        if self.nextNeural == None:
            self.dz = self.A - Y_E
            self.dw = np.dot(self.dz, self.lastNeural.A.T)/Y_E.size
            self.db = np.sum(self.dz, axis=1, keepdims=True)/Y_E.size
            self.lastNeural.Backward(Y_E)
        elif self.lastNeural != None:
            self.dz = np.dot(self.nextNeural.W.T,
                             self.nextNeural.dz) * dReLU(self.Z)
            self.dw = np.dot(self.dz, self.lastNeural.A.T)/Y_E.size
            self.db = np.sum(self.dz, axis=1, keepdims=True)/Y_E.size
            self.lastNeural.Backward(Y_E)

    def ReviseWB(self, sstep):

        if self.lastNeural == None:
            self.nextNeural.ReviseWB(sstep)
            return

        #self.W = self.W - sstep * self.dw
        #self.B = self.B - sstep * self.db
        self.vdw = beta1 * self.vdw + (1 - beta1) * self.dw
        self.vdb = beta1 * self.vdb + (1 - beta1) * self.db
        self.sdw = beta2 * self.sdw + (1 - beta2) * self.dw * self.dw
        self.sdb = beta2 * self.sdb + (1 - beta2) * self.db * self.db

        self.W = self.W - sstep * (self.vdw / np.sqrt(self.sdw + 10**-8))
        self.B = self.B - sstep * (self.vdb / np.sqrt(self.sdb + 10**-8))
        if self.nextNeural != None:
            self.nextNeural.ReviseWB(sstep)

    def OutputWB(self, config):
        folder = os.path.exists("./data")
        if not folder:
            os.makedirs("./data")
        thislayer = {}
        thislayer["index"] = self.index
        thislayer["W"] = DumpNumpyData(str(self.index)+"_W", self.W)
        thislayer["B"] = DumpNumpyData(str(self.index)+"_B", self.B)
        thislayer["vdw"] = DumpNumpyData(str(self.index)+"_vdw", self.vdw)
        thislayer["vdb"] = DumpNumpyData(str(self.index)+"_vdb", self.vdb)
        thislayer["sdw"] = DumpNumpyData(str(self.index)+"_sdw", self.sdw)
        thislayer["sdb"] = DumpNumpyData(str(self.index)+"_sdb", self.sdb)
        thislayer["neuralCount"] = self.neuralCount
        if self.activeFunc == PureOutFunc:
            thislayer["activeFunc"] = "PureOutFunc"
        elif self.activeFunc == ReLU:
            thislayer["activeFunc"] = "ReLU"
        elif self.activeFunc == SigmoidFunc:
            thislayer["activeFunc"] = "SigmoidFunc"
        else:
            thislayer["activeFunc"] = "None"
        config.append(thislayer)
        if self.nextNeural != None:
            self.nextNeural.OutputWB(config)


class NeuralNetwork:
    def __init__(self):
        self.head = NeuralLayer(1, PureOutFunc). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(16, ReLU). \
            NextLayer(1, SigmoidFunc).Head()

        self.sstep = 0.03
        self.loop = True
        self.gradcheck = False

    def NetworkStudy(self,index, X, Y):
        #print("index[", index, "] X shape:", X.shape, "Y shape:", Y.shape)
        assert(X.shape == Y.shape)

        Y_H = self.head.Forward(X)
        self.head.End().Backward(Y)
        self.head.ReviseWB(self.sstep)
        
        '''
        showindex = 0
        while True:
            Y_H = self.head.Forward(X)
            J = costFunc(LossFunc(Y_H, Y))
            self.head.End().Backward(Y)
            showindex += 1

            
            if showindex == 10000:
                showindex = 0
                print("index[", index, "]cost value=",
                      J, "@[", looptimes, "*10000]")
            #time.sleep(0.03)
                if self.gradcheck:
                    self.head.GradCheck(X, Y)
                return             
            self.head.ReviseWB(self.sstep)
        '''

    def Learn(self,_X,_Y):

        Size = _X.size
        Batch_Size = 512
        _X = Normalize(_X, 1, 0.5)


        simple_X = _X.reshape(1, Size)
        simple_Y = _Y.reshape(1, Size)
        
        x_max = np.max(simple_X)
        y_max = np.max(simple_Y)
   

        simple_X = simple_X / x_max
        simple_Y = simple_Y / y_max

        row_size = int(Size / Batch_Size)

        batch_X = simple_X.reshape(row_size, Batch_Size)
        batch_Y = simple_Y.reshape(row_size, Batch_Size)

       # X = batch_X[0, ].reshape(1, Batch_Size)
       # Y = batch_Y[0, ].reshape(1, Batch_Size)


        global interaction
        i = 0
        loopindex = 0
        showindex = 0

        while self.loop:
            X = batch_X[i, ].reshape(1, Batch_Size)
            Y = batch_Y[i, ].reshape(1, Batch_Size)
            self.NetworkStudy(i, X, Y)
           
            i += 1
            if i >= row_size:
                i =0
            
            showindex += 1
            if showindex == 10000:
                showindex = 0
                loopindex += 1
                Y_H = self.head.Forward(simple_X)
                J = costFunc(LossFunc(Y_H, simple_Y))
                print("cost value=", J, "@[", loopindex, "*10000]")

            if interaction:
                control = input("input:")
                if control == "exit":
                    exit(1)
                elif control == "showpic":
                    Y_H = self.head.Forward(simple_X)
                    show_y = Y_H * y_max
                    show_x = simple_X * x_max
                    pylab.plot(show_x[0], show_y[0])
                    pylab.plot(show_x[0], simple_Y[0] * y_max)
                    pylab.show()
                    interaction = False
                elif len(control) > 2 and (control[0] == "0" and control[1] == "."):
                    f = float(control)
                    self.sstep = f
                    print("sstep:", self.sstep)
                    interaction = False
                elif control == "next":
                    i += 1
                    if i >= row_size:
                        i =0
                    interaction = False
                else:
                    interaction = False
                    print("sstep:", self.sstep)

      


if __name__ == '__main__':
    # store the original SIGINT handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    nnw = NeuralNetwork()
    x = np.linspace(-10, 10, 2048) 
    y = SampleFunc(x) 
   
    nnw.Learn(x,y)
