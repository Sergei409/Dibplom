import time
import matplotlib.pyplot as plt
import numpy as np
from mpmath import besselj

# from scipy.spatial import distance_matrix
import scipy.spatial as sp

def rhs_point_vortex(x, y, w):
    n = np.size(x)
    resx = np.zeros(n)
    resy = np.zeros(n)

    DX = np.array([x] * n)
    DY = np.array([y] * n)
    for i in range(n):
        DX[i] = x[i] - DX[i]
        DY[i] = y[i] - DY[i]

    r = np.einsum('ji',(x, y))
    distance = sp.distance_matrix(r, r)**2 + 0.001
    np.einsum('ii->i', distance)[:] = 1

    buff_X = DX/distance
    buff_Y = DY/distance
    

    resx = np.einsum('ij,i->j', buff_Y, w)/(2*np.pi)
    resy = np.einsum('ij,i->j', buff_X, w)/-(2*np.pi)
    return resx, resy

def rungekutta4(FunFcn, x0, y0, h, W):
    s1_x , s1_y = FunFcn(x0, y0, W)
    s2_x , s2_y = FunFcn(x0 + h*s1_x/2, y0 + h*s1_y/2, W)
    s3_x , s3_y = FunFcn(x0 + h*s2_x/2, y0 + h*s2_y/2, W)
    s4_x , s4_y = FunFcn(x0 + h*s3_x, y0 + h*s3_y, W)
    x1 = x0 + h*(s1_x + 2*s2_x + 2*s3_x +s4_x)/6
    y1 = y0 + h*(s1_y + 2*s2_y + 2*s3_y +s4_y)/6
    return x1, y1

def rungekutta6(FunFcn, x0, y0, h, W):
    s1_x , s1_y = FunFcn(x0, y0, W)
    s2_x , s2_y = FunFcn(x0 + h*s1_x/3, y0 + h*s1_y/3, W)
    s3_x , s3_y = FunFcn(x0 + h*s2_x*2/3, y0 + h*s2_y*2/3, W)
    s4_x , s4_y = FunFcn(x0 + h*(s1_x/12 +s2_x/3 -s3_x/12), y0 + h*(s1_y/12 +s2_y/3 -s3_y/12), W)
    s5_x , s5_y = FunFcn(x0 + h*(s1_x*25/48-s2_x*55/24+s3_x*35/48+s4_x*15/8), y0 + h*(s1_y*25/48-s2_y*55/24+s3_y*35/48+s4_y*15/8), W)
    s6_x , s6_y = FunFcn(x0 + h*(s1_x*3/20-s2_x*11/20-s3_x/8+s4_x/2+s5_x/10), y0 + h*(s1_y*3/20-s2_y*11/20-s3_y/8+s4_y/2+s5_y/10), W)
    s7_x , s7_y = FunFcn(x0 + h*(-s1_x*261/260+s2_x*33/13+s3_x*43/156-s4_x*118/39+s5_x*32/195+s6_x*80/39), y0 + h*(-s1_y*261/260+s2_y*33/13+s3_y*43/156-s4_y*118/39+s5_y*32/195+s6_y*80/39), W)
    x1 = x0 + h*(13*s1_x + 55*s3_x + 55*s4_x + 32*s5_x + 32*s6_x + 13*s7_x)/200
    y1 = y0 + h*(13*s1_y + 55*s3_y + 55*s4_y + 32*s5_y + 32*s6_y + 13*s7_y)/200
    return x1, y1

def count(val_left, val_right, h, R):
    k = 0

    for y in np.arange(val_left, val_right, h):
        for x in np.arange(val_left, val_right, h):
            r = np.sqrt(x**2 + y**2)
            if r < R:
                k += 1  
    
    return k

def main():
    R = 0.5         # Радиус диполя
    Uvel = 1                            # Скорость перемещения диполя
    x = np.arange(0, 5, 0.01)
    y = np.i0(x)
    lam = 3.8317 / R                       

    val_left = -1.1
    val_right = 1.1
    n = 99
    h = (val_right - val_left) / n
    ht = 0.05                            # Шаг прохода

    k = count(val_left, val_right, h, R)
    print(k)
    X = np.zeros(2*k)
    Y = np.zeros(2*k)
    W = np.zeros(2*k)
    cnt = 0

    # Инициализация начальных положений точек 
    # и коэффициентов завихренности

    for y in np.arange(val_left, val_right, h):
        for x in np.arange(val_left, val_right, h):
            r = np.sqrt(x**2 + y**2)
            if r < R:
                X[cnt] = x
                X[k+cnt] = x
                Y[cnt] = y
                Y[k+cnt] = y + 2.0
                phi = np.arctan2(x, y)
                W[cnt] = (2*lam*Uvel*besselj(1, lam*r)/besselj(0, lam*R)*np.sin(phi)*h**2)
                W[k + cnt] = (2*lam*(-Uvel)*besselj(1, lam*r)/besselj(0, lam*R)*np.sin(phi)*h**2)
                cnt += 1

    # Построение графика
    delmn = min(W)
    delmx = max(W)

    colors2 = []
    for i in range(2 * k):
        colors2.append([0.5,(W[i]-delmn)/(delmx-delmn),(W[i]-delmn)/(delmx-delmn)])

    plt.ion()
    #plt.figure(figsize=(5, 5), dpi=50)
    #plt.xlim([-0.2, 0.2])
    #plt.ylim(-2, 2)
    #plt.rcParams['figure.figsize'] = [2, 4]
    for delay in np.arange(0, 2, ht):
            #start = time.time()
            X, Y = rungekutta4(rhs_point_vortex, X, Y, ht, W)
            #end = time.time()
            #print("The time of execution of above program is :", (end-start) * 10**3, "ms")

            plt.clf()
            plt.scatter(X, Y,s=50 ,c=colors2)
            plt.suptitle('t=%3.2f' % (delay + ht))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim([-2.0, 2.0])
            plt.ylim([-1.0, 3.0])
            plt.draw()

            #plt.pause(100)
            plt.gcf().canvas.flush_events()
            #plt.savefig("test_%3.2f.png" % (delay + ht))

            time.sleep(0.02)
            #------------------------------------------------------------

            # X, Y = rungekutta4(rhs_point_vortex, X, Y, delay, W)

    plt.ioff()
    plt.show()
    print(k)


main()