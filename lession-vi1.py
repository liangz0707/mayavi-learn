#coding=utf-8
#__author__ = 'lz'

from numpy import pi,sin,cos,mgrid
from mayavi import mlab
import numpy as np

def t_plots3d():
    dphi ,dtheta = pi/250.0,pi/250.0
    [phi ,theta] = mgrid[0:pi+dphi*1.5 :dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4 ;m1 = 3;m2 = 2;m3 = 3; m4 = 6;m5 = 2;m6=6;m7 = 4;

    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    s = mlab.mesh(x, y, z)

def t_points3d():
    t = np.linspace(0, 4*np.pi , 20)

    x = np.cos(t)
    y = np.sin(2*t)
    z = np.cos(2*t)
    s = 2+sin(t)

    mlab.points3d(x,y,z,s, colormap="copper", scale_factor=.25)

def t_plot3d():
    n_mer ,n_long =6,11
    pi = np.pi
    dphi = pi / 1000.0
    phi = np.arange(0.0, 2 * pi + 0.5 * dphi, dphi)

    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')

from scipy import misc #图像处理

def t_imshow():
    img = misc.imread('1.jpg')
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]
    print (img_G.shape,img_G.dtype)
    g = np.asarray(img_G,dtype='float32')
    print (g.shape ,g.dtype)
    mlab.imshow(g,colormap='gist_earth')

def t_surf():
    img = misc.imread('1.jpg')
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]
    print (img_G.shape,img_G.dtype)
    g = np.asarray(img_G,dtype='float32')
    print (g.shape ,g.dtype)
    mlab.surf(g,colormap='gist_earth')

def t_contour_surf():
    img = misc.imread('1.jpg')
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]
    print (img_G.shape,img_G.dtype)
    g = np.asarray(img_G,dtype='float32')
    print (g.shape ,g.dtype)
    mlab.contour_surf(g,colormap='gist_earth')

def t_mesh():
    pi = np.pi
    cos = np.cos
    sin = np.sin
    dphi, dtheta = pi / 250.0, pi / 250.0
    [phi, theta] = np.mgrid[0:pi + dphi * 1.5:dphi,
                               0:2 * pi + dtheta * 1.5:dtheta]
    m0 = 4
    m1 = 3
    m2 = 2
    m3 = 3
    m4 = 6
    m5 = 2
    m6 = 6
    m7 = 4
    r = sin(m0 * phi) ** m1 + cos(m2 * phi) ** m3 + \
        sin(m4 * theta) ** m5 + cos(m6 * theta) ** m7
    x = r * sin(phi) * cos(theta)
    y = r * cos(phi)
    z = r * sin(phi) * sin(theta)

    mlab.mesh(x, y, z, colormap="bone")

'''
t_imshow()
t_contour_surf()
t_surf()

mlab.show() #一个show()可以把之前的内容都放在同一个画面上
'''
def test_contour3d():
    #x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

    x=np.asarray([[[-1]],[[0]],[[3]],[[4]]])
    y=np.asarray([[[-1],[3],[6],[9]]])
    z=np.asarray([[[-3,-2.5,3.8,4.9]]])
    scalars = x * 0.5 + y * y + z * z
    print scalars.shape
    obj = mlab.contour3d(scalars, contours=[1,4,6,9,10], transparent=True)
    return obj

def test_flow():
    x, y, z = np.mgrid[0:5, 0:5, 0:5]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / r
    v = -x * np.sin(r) / r
    w = np.zeros_like(z)
    obj = mlab.flow(x,y,z+10,u, v, w)
    print x.shape ,y.shape,u.shape,y.shape
    return obj

def t_valum():
    x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    s = np.sin(x*y*z)/(x*y*z)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=0, vmax=0.8)

t_valum()
mlab.show()