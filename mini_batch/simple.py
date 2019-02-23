import signal
import time
import sys
import numpy as np
import pylab 
import yaml
import os



       
Size = 1024 
sstep = 0.01
loop = True
interaction = False

#sigma = 0.4
#mu = 0


    
           
def run_program():
    #pylab.plot(x , y )
    #pylab.show()
    loop = True
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

        n3 =   NeuralLayer(3,20,ReLU)
        n3.SetLastNeural(n2)

        n4 =   NeuralLayer(4,20,ReLU)
        n4.SetLastNeural(n3)

        n5 =   NeuralLayer(5,20,ReLU)
        n5.SetLastNeural(n4)

        n6 =   NeuralLayer(6,20,ReLU)
        n6.SetLastNeural(n5)

        n7 =   NeuralLayer(7,1,SigmoidFunc)
        n7.SetLastNeural(n6)
        
        last_nu = n7
  

        x=  np.linspace(0,100,Size)    
        y = SampleFunc(x)

    

    Y_H = 0


    

  #  pylab.plot(x , y )
  #  pylab.show()
    #print(x[0:Batch_Size])
    #X = batch_X[0, ].reshape(1, Batch_Size) / Batch_Size
    #Y = batch_Y[0, ].reshape(1, Batch_Size) / Batch_Size
    simple_X = x.reshape(1, Size)
    simple_Y = y.reshape(1, Size)

    Batch_Size = 128
    row_size = int(Size / Batch_Size)

    batch_X = simple_X.reshape(row_size, Batch_Size)
    batch_Y = simple_Y.reshape(row_size, Batch_Size)

    X = batch_X[0, ].reshape(1, Batch_Size)
    Y = batch_Y[0, ].reshape(1, Batch_Size)


    x_max = np.max(X.reshape(1, Batch_Size))
    x_min = np.min(X.reshape(1, Batch_Size))


    y_max = np.max(Y.reshape(1, Batch_Size))
    y_min = np.min(Y.reshape(1, Batch_Size))

    print("x max:", x_max, "y max:", y_max)

    X = x_min + X / x_max
    Y = y_min + Y / y_max

    gradcheck = False
    showindex = 0 


    print("X shape:", X.shape, "Y shape:", Y.shape)
  

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
            if J < 0.1:
                break
            if interaction:
                control = input("input:")
                if control=="exit":
                    loop = False
                elif control=="showpic":
                    x_simple = X[0]
                    y_simple = Y[0]
                    
                    pylab.plot((x_simple * x_max), (Y_H * y_max)[0])
                    pylab.plot((x_simple * x_max), (y_simple * y_max))
                    pylab.show()
                elif control=="showtest":
                    '''
                    test_size = 100
                    test_x=  np.linspace(0,20,test_size)  
                    test_y = SampleFunc(test_x)
                    test_X = test_x.reshape(1,test_size)/ test_size
                    test_Y = n0.Output(test_X)
                    show_test_y = test_Y * test_size
                    pylab.plot(test_x , show_test_y[0] )
                    pylab.plot(test_x , test_y )
                    pylab.show()
                    '''
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



if __name__ == '__main__':
    # store the original SIGINT handler


    run_program()
