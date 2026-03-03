#!/bin/env python
#Inspired from https://github.com/wangsl/python-embedding
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def py_torch2(x1) :
    """This is a more expensive calculation than py_torch1, which we can use to test the non-blocking behavior."""
    #x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1))

    print("py_torch2: Shape of the input array x : ", x1.shape)
    print("py_torch2: sum(x1) ", a)
    sys.stdout.flush()  #do this after all print statements
    
    #The input array from fortran has the wrong, reversed shape
    #E.e., an input array of fortran extents of (i=1:4, j=1:3) gets a python shape of (4,3).
    #But, it has to be 3by4 matrix of shape (3,4)
    #We have to reshape the input array before doing anything useful with it.
    #
    pyx=x1.reshape(x1.shape[::-1]) #same as with order='C'
    print("py_torch2: Shape of the reshaped input array pyx : ", pyx.shape)
    #  
    #We could do an expensive calculation here and modify the input array x1 
    #x1[:,:] = x1 * 10.

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA (GPU) is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    sys.stdout.flush()  #do this after all print statements
    
    data_cpu = torch.from_numpy(pyx)
    print("Data converted to torch tensor on CPU. shape= {}, sum= {}".format(data_cpu.shape, torch.sum(data_cpu)));sys.stdout.flush()
    data_gpu = data_cpu.to(device)
    print("Data moved to device {}. shape= {}, sum= {}".format(device, data_gpu.shape, torch.sum(data_gpu)));sys.stdout.flush()
    ## cov_matrix_gpu = torch.cov(data_gpu) #covariance results a square matrix, so not what we want here
    
    #do some silly expensive calculation for testing purposes. We can replace this with a real calculation later. The point is to make it expensive enough to test the non-blocking behavior of the plot.
    # Create a 200x180 tensor and a large 180x10000 tensor
    print("Starting expensive GPU calculation on device {}...".format(device));sys.stdout.flush()
    A = torch.randn(10000, 100000)
    B = torch.randn(100000, 10000)
    # GPU Execution (Moving data to GPU first)
    A_gpu, B_gpu = A.to(device), B.to(device)
    # consumes 8746MiB of GPU memory and takes about 1 minute on V100
    for i in range(50): #repeat a few times to make it more expensive
        result_gpu = torch.matmul(A_gpu, B_gpu)
    gpusum = torch.sum(result_gpu).item() # Do something with the result to ensure the computation is performed
    torch.cuda.synchronize() # Wait for GPU to finish
    print("Result of expensive GPU calculation: ", gpusum);sys.stdout.flush()      
    #To make the calculation more expensive do this a frew times in a loop
    for i in range(1000):
        #DOWNSAMPLE: Reduce by factor of 2
        cov_matrix_gpu = F.interpolate(data_gpu.unsqueeze(0).unsqueeze(0), scale_factor=0.5,mode='bilinear', align_corners=False).squeeze(0)
        #print("Downsampled on device {}. shape= {}, sum= {}".format(device, cov_matrix_gpu.shape, torch.sum(cov_matrix_gpu)));sys.stdout.flush()
        #UPSAMPLE / REPEAT: Back to origial shape by repeating cells
        # 'nearest' mode effectively "repeats" the pixel values
        repeated = F.interpolate(cov_matrix_gpu.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
        #print("Upsampled on device {}. shape= {}, sum= {}".format(device, repeated.shape, torch.sum(repeated)));sys.stdout.flush()
        cov_matrix_cpu = repeated.to("cpu")

    print("Covariance matrix computed on device {}. shape= {}, sum= {}".format(torch.device, cov_matrix_cpu.shape, torch.sum(cov_matrix_cpu)));sys.stdout.flush()
    #sys.stdout.flush()  #do this after all print statements
    cov_matrix_cpu_np = cov_matrix_cpu.numpy()
    #print("Covariance matrix computed on device {}. shape= {}, sum= {}".format(device, cov_matrix_cpu_np.shape, np.sum(cov_matrix_cpu_np)));sys.stdout.flush()
    #print("Need to reshape the covariance matrix back to the original shape of the input array x1, which is {}".format(x1.shape));sys.stdout.flush()
    #sys.stdout.flush()  #do this after all print statements 
    
    x1[:,:] = cov_matrix_cpu_np.reshape(x1.shape) #reshape back to the original shape of the input array x1, which is (4,3) in this example
    
    sys.stdout.flush()  #do this after all print statements
    return a

def py_torch1(x1) :
    #x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1))

    print("py_torch1: Shape of the input array x : ", x1.shape)
    print("py_torch1: sum(x1) ", a)
    sys.stdout.flush()  #do this after all print statements
    
    #The input array from fortran has the wrong, reversed shape
    #E.e., an input array of fortran extents of (i=1:4, j=1:3) gets a python shape of (4,3).
    #But, it has to be 3by4 matrix of shape (3,4)
    #We have to reshape the input array before doing anything useful with it.
    #
    pyx=x1.reshape(x1.shape[::-1]) #same as with order='C'
    print("py_torch1: Shape of the reshaped input array pyx : ", pyx.shape)
    #  
    #We could do an expensive calculation here and modify the input array x1 
    #x1[:,:] = x1 * 10.

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA (GPU) is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    sys.stdout.flush()  #do this after all print statements
    
    data_cpu = torch.from_numpy(pyx)
    print("Data converted to torch tensor on CPU. shape= {}, sum= {}".format(data_cpu.shape, torch.sum(data_cpu)));sys.stdout.flush()
    data_gpu = data_cpu.to(device)
    print("Data moved to device {}. shape= {}, sum= {}".format(device, data_gpu.shape, torch.sum(data_gpu)));sys.stdout.flush()
    ## cov_matrix_gpu = torch.cov(data_gpu) #covariance results a square matrix, so not what we want here
    #DOWNSAMPLE: Reduce by factor of 2
    cov_matrix_gpu = F.interpolate(data_gpu.unsqueeze(0).unsqueeze(0), scale_factor=0.5,mode='bilinear', align_corners=False).squeeze(0)
    print("Downsampled on device {}. shape= {}, sum= {}".format(device, cov_matrix_gpu.shape, torch.sum(cov_matrix_gpu)));sys.stdout.flush()
    #UPSAMPLE / REPEAT: Back to origial shape by repeating cells
    # 'nearest' mode effectively "repeats" the pixel values
    repeated = F.interpolate(cov_matrix_gpu.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
    print("Upsampled on device {}. shape= {}, sum= {}".format(device, repeated.shape, torch.sum(repeated)));sys.stdout.flush()
    cov_matrix_cpu = repeated.to("cpu")
    print("Covariance matrix computed on device {}. shape= {}, sum= {}".format(device, cov_matrix_cpu.shape, torch.sum(cov_matrix_cpu)))
    sys.stdout.flush()  #do this after all print statements
    cov_matrix_cpu_np = cov_matrix_cpu.numpy()
    print("Covariance matrix computed on device {}. shape= {}, sum= {}".format(device, cov_matrix_cpu_np.shape, np.sum(cov_matrix_cpu_np)));sys.stdout.flush()
    print("Need to reshape the covariance matrix back to the original shape of the input array x1, which is {}".format(x1.shape));sys.stdout.flush()
    sys.stdout.flush()  #do this after all print statements
    x1[:,:] = cov_matrix_cpu_np.reshape(x1.shape) #reshape back to the original shape of the input array x1, which is (4,3) in this example
    
    sys.stdout.flush()  #do this after all print statements
    return a


def py_plot2Darrays(x1) :
    # 
    #x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1))

    print("py_plot2Darrays: Shape of the input array x : ", x1.shape)
    print("py_plot2Darrays: sum(x1) ", a)
    sys.stdout.flush()  #do this after all print statements
    
    #The input array from fortran has the wrong, reversed shape
    #E.e., an input array of fortran extents of (i=1:4, j=1:3) gets a python shape of (4,3).
    #But, it has to be 3by4 matrix of shape (3,4)
    #We have to reshape the input array before doing anything useful with it.
    #
    pyx=x1.reshape(x1.shape[::-1]) #same as with order='C'
    #
    #Plot the input field
    myplot(pyx)
   
    #We could do an expensive calculation here and modify the input array x1 
    #x1[:,:] = 10.

    sys.stdout.flush()  #do this after all print statements
    return a

def myplot(x):   
    #Plot the input array
    fig, ax = plt.subplots()
    plt.pcolormesh(x[:,:]);plt.colorbar()
    #These are all non-blocking, but they make a new figure per call 
    #none of them plots on the same canvas
    plt.show(block=False)
    plt.pause(0.1)
    #Or
    #plt.draw()
    #plt.pause(0.01) # This handles the GUI events to keep it non-blocking
    #Or
    # Force a redraw and flush events
    #plt.show(block=False)
    #fig.canvas.draw()
    #fig.canvas.flush_events()

def py_plot2Darrays_blocking(x1) :
    #return 
    print("py_plot2Darrays_blocking: Shape of the input array x : ", x1.shape)
    x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1))
    print("py_plot2Darrays_blocking: sum(x1) ", a)
    #The input array from fortran has the wrong, reversed shape
    #E.e., an input array of fortran extents of (i=1:4, j=1:3) gets a python shape of (4,3).
    #But, it has to be 3by4 matrix of shape (3,4)
    #We have to reshape the input array before doing anything useful with it.
    #
    pyx=x1.reshape(x1.shape[::-1]) #same as with order='C'
    #

    sys.stdout.flush()  #do this after all print statements

    #Plot the input array
    fig, ax = plt.subplots()
    plt.pcolormesh(pyx[:,:]);plt.colorbar()
    plt.show()

    return a

def py_plot2Darrays0(x1) :
    #return 
    print("py_plot2Darrays: Shape of the input array x : ", x1.shape)
    x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1))
    print("py_plot2Darrays: sum(x1) ", a)
    #print(x1)
#for an input array of fortran shape (i=1:4, j=1:3) made from tmp2d(i,j) = i + (j-1)*4
#we get x1 to be wrong shape:
#[[ 1.  2.  3.]
# [ 4.  5.  6.]
# [ 7.  8.  9.]
# [10. 11. 12.]]
    #xf=x1.flatten()
#[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]    
    #print("xf.shape and x1.shape",xf.shape, x1.shape)
#xf.shape and x1.shape (12,) (4, 3)    
    print("index where x1=6. ",  np.where(x1 == 6.0))    
    #print("index where x1.flatten=2. ",  np.where(xf == 6.0))

#To fix this issue we can reshape the flattened array to (3,4). Is there a better/more succint way?    
    #print("py_plot2Darrays: reshape F ")
    #pyx=xf.reshape(x1.shape[::-1], order='F') #wrong results:
#[[ 1.  4.  7. 10.]
# [ 2.  5.  8. 11.]
# [ 3.  6.  9. 12.]]
    print("py_plot2Darrays: reshape C ") 
    #pyx=x1.reshape(x1.shape[::-1], order='C')
    pyx=x1.reshape(x1.shape[::-1]) #same as with order='C'
    #pyx=xf.reshape(x1.shape[::-1]) #same as x1.reshape , so not necessary to flatten 
    #print(pyx)
    print("index where pyx=6 ",  np.where(pyx == 6.0))
#[[ 1.  2.  3.  4.]
# [ 5.  6.  7.  8.]
# [ 9. 10. 11. 12.]]
#index where pyx=6  (array([1]), array([1]))    
    sys.stdout.flush()

    #Plot the input array
    fig, ax = plt.subplots()
    plt.pcolormesh(pyx[:,:]);plt.colorbar()
    plt.show(block=False)
    #plt.pause(0.1)
    #Or
    # Force a redraw and flush events
    fig.canvas.draw()
    fig.canvas.flush_events()
    #What happens if we modify the input array?
    #It changes the input array in the fortran caller. So we can modify it here and pass it back.
    #We could do an expensive calculation here and modify x1
    #x1[:,:] = 10.

    return a

def py_plot3Darrays(x1) :
    #return 
    print("py_plot3Darrays: Shape of the input array x : ", x1.shape)
    x1[np.isnan(x1)] = 0
    a=np.sum(x1,axis=(0,1,2))
    print("py_plot3Darrays: sum(x1) ", a)
    pyx=x1[:,:,1]
    print("pyx.shape ",pyx.shape)
    print("pyx[120,90] ",pyx[120,90])
    print("pyx[90,120] ",pyx[90,120])
    #pyx=pyx.reshape(pys.shape, order='F')
    sys.stdout.flush()
    plt.pcolormesh(pyx[:,:])
    plt.show();
    return a

def py_plot1Darrays0(x1,x2) :
    #return 
    #print("py_plot1Darrays: Shape of the input array x : ", x1.shape, x2.shape)
    a=np.sum(x1[:])
    print("py_plot1Darrays: sum(x1) ", a)
    print("py_plot1Darrays: sum(x2) ", np.sum(x2))
    sys.stdout.flush()
    plt.plot(x1[:],marker='o',color='r',label='array 1')
    plt.plot(x2[:],marker='*',color='b',label='array 2')
    plt.legend(loc='upper right')
    plt.show();
    return a

def py_plot1Darrays(x1,x2,str1=None) :
    #return 
    #print("py_plot1Darrays: Shape of the input array x : ", x1.shape, x2.shape)
    a=np.sum(x1[:])
    print("py_plot1Darrays: sum(x1) ", a)
    print("py_plot1Darrays: sum(x2) ", np.sum(x2))
    if str1 is not None:
       parts=str1.split(",") 
    sys.stdout.flush()
    plt.title(parts[0])
    plt.plot(x1[:],marker='o',color='r',label='array 1, '+parts[1])
    plt.plot(x2[:],marker='*',color='b',label='array 2, '+parts[2])
    plt.legend(loc='upper right')
    plt.show();
    return a

