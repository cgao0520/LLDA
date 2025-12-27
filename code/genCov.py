import numpy as np

def genCov(img):
    '''
    genCov(img) function generates covariance feature for image img
    img need to be numpy.array type
    '''
    r,c=img.shape

    Fx,Fy = np.gradient(img)
    Fxx,Fxy = np.gradient(Fx)
    Fyx,Fyy = np.gradient(Fy)
    
    aFx = np.abs(Fx)
    aFy = np.abs(Fy)
    aFxx = np.abs(Fxx)
    aFxy = np.abs(Fxy)
    aFyx = np.abs(Fyx)
    aFyy = np.abs(Fyy)

    m=[]

    for y in range(r):
        for x in range(c):
            at = np.arctan(aFx[y,x]/np.max((aFy[y,x],0.0000001)))
            px = [y,x,img[y,x],aFx[y,x],aFy[y,x],aFxx[y,x],aFxy[y,x],aFyx[y,x],aFyy[y,x],at]
            m.append(px)
    
    return np.cov(np.array(m).T)
