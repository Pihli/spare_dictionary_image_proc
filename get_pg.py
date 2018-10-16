# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 00:11:53 2018

@author: Touch
"""

def Get_PG( im,win, ps ,nlsp,step,delta):
#function   [Px, Px0] =  Get_PG( im,win, ps ,nlsp,step,delta)

    h, w, _  =  im.size()
    S         =  win
    maxr         =  h-ps+1
    maxc         =  w-ps+1
    r         =  range(maxr,step)
    r         =  [r] + range(r[-1],maxr)
    c         =  range(maxc,step)
    c         =  [c] + range(c[-1],maxc)
    X = np.zeros([ps**2,maxr*maxc],np.float32)
    Px0 = []
    Px = []
    # TODO
    if nlsp == 1:
        k    =  0
        for i in range(ps):
            for j in range(ps):
                k    +=  1
                blk     =  im[r-1+i,c-1+j]
                Px(k,:) =  blk(:)'
    else
        k    =  0
        for i  = 1:ps
            for j  = 1:ps
                k    =  k+1
                blk  =  im(i:end-ps+i,j:end-ps+j)
                X(k,:) =  blk(:)'
        # Index image
        Index     =   (1:maxr*maxc)
        Index    =   reshape(Index, maxr, maxc)
        N1    =   length(r)
        M1    =   length(c)
        blk_arr   =  zeros(nlsp, N1*M1 )
        for  i  =  1 :N1
            for  j  =  1 : M1
                row     =   r(i)
                col     =   c(j)
                off     =  (col-1)*maxr + row
                off1    =  (j-1)*N1 + i
                
                rmin    =   max( row-S, 1 )
                rmax    =   min( row+S, maxr )
                cmin    =   max( col-S, 1 )
                cmax    =   min( col+S, maxc )
                
                idx     =   Index(rmin:rmax, cmin:cmax)
                idx     =   idx(:)
                neighbor       =   X(:,idx)
                seed       =   X(:,off)
                
                dis     =   (neighbor(1,:) - seed(1)).^2
                for k = 2:ps^2
                    dis   =  dis + (neighbor(k,:) - seed(k))**2
                end
                dis = dis./ps**2
                [~,ind]   =  sort(dis)
                indc        =  idx( ind( 1 : nlsp ) )
                blk_arr(:,off1)  =  indc
                X_nl = X(:,indc); % or X_nl = neighbor(:,ind( 1 : nlsp ))
                # Removes DC component from image patch group
                DC = mean(X_nl,2)
                X_nl = bsxfun(@minus, X_nl, DC)
                # Select the smooth patches
                sv=var(X_nl)
                if max(sv) <= delta:
                    Px0 = [Px0 X_nl]
                else:
                    Px = [Px X_nl]
        return Px, Px0