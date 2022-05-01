# -*- coding: utf-8 -*-
#author:Yan Mi
#Start data:2021.5.12
#End data:2021.5.12
'''
I calculate one dimension Heisenberg model with spin 3
'''

import numpy as np

#精度和时间间隔
eps = 0.0000000001
deltat = 0.01
maxtimes = 10000

def beautifulPrintMatrix(A):
    n = A.shape[0]
    m = A.shape[1]
    for i in range(0,n):
        for j in range(0,m):
            oe = A[i,j]
            print("%.4f + %.1fi "%(oe.real,oe.imag),end="  ")
        print("")
        
def getSeoMatrix():
    ts = 2 ** 0.5
    sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    sx = np.array([[0,ts/2,0],[ts/2,0,ts/2],[0,ts/2,0]])
    sy = np.array([[0,ts,0],[-ts,0,ts],[0,-ts,0]])/2j
    res = np.kron(sz,sz) + np.kron(sx,sx) + np.kron(sy,sy)
    return np.real(res)

# beautifulPrintMatrix(getSeoMatrix())

class Solution():
    def __init__(self,r,precision,deltat,maxtimes):
        self.deltat = deltat
        self.r = r
        self.maxtimes = maxtimes
        self.precision = precision
        self.h = getSeoMatrix()
        self.sigma=np.random.rand(2,self.r)
        self.g=np.random.rand(2,self.r,3,self.r)
    
    def getTimeOperator(self):
        [va,ve] = np.linalg.eig(self.h)
        '''u = exp(-th)'''
        u = self.u = np.array(np.mat(ve)*np.diag(np.exp(-self.deltat*va))*np.mat(ve).H)
        self.u = np.reshape(self.u,(3,3,3,3))
        return u
    
    def itebd(self):
        self.getTimeOperator()
        # E1 = 0
        E2 = 0
        Energy = 2
        times = 0
        [A,B] = [1,0]
        # while(abs(Energy - E1) > self.precision):
        while(times < self.maxtimes):
            [A,B] = [B,A]
            Psi = np.tensordot(np.diag(self.sigma[B,:]),self.g[A,...],axes=(1,0))
            Psi = np.tensordot(Psi,np.diag(self.sigma[A,:]),axes=(2,0))
            Psi = np.tensordot(Psi,self.g[B,...],axes=(2,0))
            Psi = np.tensordot(Psi,np.diag(self.sigma[B,:]),axes=(3,0))

    
            Psi = np.tensordot(Psi,self.u,axes=((1,2),(0,1)))
            
            #分解
            Psi = np.reshape(np.transpose(Psi,(0,2,3,1)),(self.r*3,3*self.r))
            [Left,newsigma,Right] = np.linalg.svd(Psi)
            
            #归一化
            self.sigma[A,:] = newsigma[0:self.r]/np.sqrt(np.sum(newsigma[0:self.r]**2))

            #指标断开
            Left = np.reshape(Left[0:3*self.r,0:self.r],(self.r,3,self.r))
            self.g[A,...] = np.tensordot(np.diag(self.sigma[B,:]**(-1)),Left,axes=(1,0))
    
            Right = np.reshape(Right[0:self.r,0:3*self.r],(self.r,3,self.r))
            self.g[B,...] = np.tensordot(Right,np.diag(self.sigma[B,:]**(-1)),axes=(2,0))

            # E1 = E2
            [E2,Energy] = [Energy,-np.log(np.sum(Psi**2))/(self.deltat*2)]
          
            if(times == self.maxtimes // 12):
                print("迭代次数为%d时的能量为%.12f"%(times,(Energy+E2)/2))
            elif(times == self.maxtimes // 2):
                print("迭代次数为%d时的能量为%.12f"%(times,(Energy+E2)/2))
            elif(times == 100):
                print("迭代次数为%d时的能量为%.12f"%(times,(Energy+E2)/2))
            
            times += 1
        return (Energy + E2)/2         
        
test = Solution(30,eps,deltat,maxtimes)
print("迭代次数为%d时的能量为%.12f"%(maxtimes,test.itebd()))
