# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 00:11:53 2018

@author: Touch
"""
import numpy as np

def Get_PG( im,win, ps ,nlsp,step,delta):

    h, w  =  im.shape[:2]
    S         =  win
    maxr         =  h-ps+1
    maxc         =  w-ps+1
    r         =  np.arange(0,maxr,step)
    r         =  np.append(r, maxr-1)
#    r         =  np.append(r, np.arange(r[-1],maxr))
    c         =  np.arange(0,maxc,step)
    c         =  np.append(c, maxc-1)
#    c         =  np.append(c, range(c[-1],maxc))
    X = []
    Px0 = []
    Px = []
    # TODO
    if nlsp == 1:
        for i in range(ps):
            imr = im[r+i]
            for j in range(ps):
                blk = imr[:,c+j]
                Px.append(blk.flatten())
        Px = np.array(Px)
    else:
        for i in range(ps):
            imr = im[i:h-ps+i+1]
            for j  in range(ps):
                blk  =  imr[:,j:w-ps+j+1]
                X.append(blk.flatten())
        X = np.array(X,np.float32)
        # Index image
        Index     =   np.arange(maxr*maxc)
        Index.resize([maxr, maxc])
        N1    =   len(r)
        M1    =   len(c)
        blk_arr   =  np.zeros([nlsp, N1*M1])
        for  i  in range(N1):
            for  j in range(M1):
                row     =   r[i]
                col     =   c[j]
                off     =  col + row*maxc
                off1    =  j + i*M1
                
                rmin    =   max( [row-S, 0] )
                rmax    =   min( [row+S, maxr] )
                cmin    =   max( [col-S, 0] )
                cmax    =   min( [col+S, maxc] )
                
                idx     =   Index[rmin:rmax, cmin:cmax]
                idx     =   idx.flatten()
                neighbor       =   X[:,idx]
                seed       =   X[:,off].reshape([-1,1])
                
                dis = abs(neighbor - seed)**2
                dis = dis.mean(axis=0)
                ind   =  np.argsort(dis)
                indc = idx[ind[:nlsp]]
                blk_arr[:,off1]  =  indc
                X_nl = X[:,indc]
                # or X_nl = neighbor(:,ind( 1 : nlsp ))
                # Removes DC component from image patch group
                DC = X_nl.mean(axis=1).reshape([-1,1])
                X_nl = X_nl - DC
                # Select the smooth patches
                sv = X_nl.var(axis=0)
                if sv.max() <= delta:
                    Px0.append(X_nl)
                else:
                    Px.append(X_nl)
    return np.concatenate(Px,axis=1), np.concatenate(Px0,axis=1)