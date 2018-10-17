# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 00:11:53 2018

@author: Touch
"""

def Get_PG( im,win, ps ,nlsp,step,delta):
#function   [Px, Px0] =  Get_PG( im,win, ps ,nlsp,step,delta)

    h, w  =  im.shape()[:2]
    S         =  win
    maxr         =  h-ps+1
    maxc         =  w-ps+1
    r         =  np.arange(0,maxr,step)
    r         =  np.append([r], np.arange(0,r[-1],maxr))
    c         =  np.arange(0,maxc,step)
    c         =  np.append([c], range(0,c[-1],maxc))
    X = np.zeros([ps**2,maxr*maxc],np.float32)
    Px0 = []
    Px = []
    # TODO
    if nlsp == 1:
        for i in range(ps):
            for j in range(ps):
                blk = im[r+i][:,c+j]
                Px = np.append(Px,blk.reshape([1,-1]),axis=0)
    else:
        for i in range(ps):
            for j  in range(ps):
                blk  =  im[i:maxr-ps+i][:,j:maxc-ps+j]
                X = np.append(X,blk.reshape([1,-1]),axis=0)
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
                idx     =   idx.reshape([-1])
                neighbor       =   X[:,idx]
                seed       =   X[:,off]
                
                dis     =   (neighbor[0,:] - seed[0])**2
                for k in range(1,ps*ps):
                    dis  +=  (neighbor[k,:] - seed[k])**2
                dis = dis/ps**2
                ind   =  np.argsort(dis)
                indc = idx[ind[:nlsp]]
                blk_arr[:,off1]  =  indc
                X_nl = X[:,indc]
                # or X_nl = neighbor(:,ind( 1 : nlsp ))
                # Removes DC component from image patch group
                DC = X_nl.mean(1)
                X_nl = bsxfun(@minus, X_nl, DC)
                # Select the smooth patches
                sv=var(X_nl)
                if max(sv) <= delta:
                    Px0 = [Px0 X_nl]
                else:
                    Px = [Px X_nl]
        return Px, Px0