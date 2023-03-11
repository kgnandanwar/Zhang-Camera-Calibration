#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import scipy
from timeit import default_timer as timer


def error(intrinsic_list, Extrinsic_mat, lamb_mat, objpoints, imgpoints, flag =0):
    er_mat = []
    H_mat = np.zeros([3,3])
    #initial_params = (alpha, gamma, u , beta, v)
    alpha_opt = intrinsic_list[0]
    gamma_opt = intrinsic_list[1]
    u_opt = intrinsic_list[2]
    beta_opt = intrinsic_list[3]
    v_opt = intrinsic_list[4]
    k1_opt = intrinsic_list[5]
    k2_opt = intrinsic_list[6]
    intrinsic_opt = np.zeros([3,3])
    intrinsic_opt[0][0] = alpha_opt
    intrinsic_opt[0][1] = gamma_opt
    intrinsic_opt[0][2] = u_opt
    intrinsic_opt[1][1] = beta_opt
    intrinsic_opt[1][2] = v_opt
    intrinsic_opt[2][2] = 1
    er_sum = 0
    for i in range(len(Extrinsic_mat)):
        img = cv.imread(images[i])
        r1 = Extrinsic_mat[i][:,0]
        r2 = Extrinsic_mat[i][:,1]
        t = Extrinsic_mat[i][:,3]
        H_mat[:,0] = r1
        H_mat[:,1] = r2
        H_mat[:,2] = t
        al = np.dot(intrinsic_opt, H_mat)
        er_sum = 0
        #print(H_mat)
        for j in range(54):            
            M = objpoints[i][j][0], objpoints[i][j][1]
            M = np.asarray(M)
            M = np.append(M,1)
            M1 = np.dot(al,M)
            #M1 = M1/lamb_mat[12]
            M1[0] = M1[0]/lamb_mat[i]
            M1[1] = M1[1]/lamb_mat[i] 
            M1[2] = M1[2]/lamb_mat[i]
            M1[0] = M1[0]/M1[2]
            M1[1] = M1[1]/M1[2]
            M1[2] = M1[2]/M1[2]
            x = M1[0].copy()
            y = M1[1].copy()
            x_dash = x + (x-u)*(k1_opt*(x**2 + y**2) + k2_opt*((x**2 + y**2)**2))
            y_dash = y + (y-v)*(k1_opt*(x**2 + y**2) + k2_opt*((x**2 + y**2)**2))
            M_proj = np.zeros_like(M)
            M_proj[0] = x_dash
            M_proj[1] = y_dash
            M_proj[2] = 1
            M_proj = np.asarray(M_proj, dtype=int)
            #print(int(M_proj[0]))
            
            if flag : 
                cv.circle(img, (int(M_proj[0]),int(M_proj[1])), 8, (0, 0, 255), -1)

            m = imgpoints[i][j]
            m = [imgpoints[i][j][0][0], imgpoints[i][j][0][1], 1]
            m = [imgpoints[i][j][0][0], imgpoints[i][j][0][1], 1]
            er = m-M_proj
            er = ((er[0])**2+(er[1])**2+(er[2])**2)
            er_sum = er_sum + er
        er_sum1 = er_sum/54
        if flag :
            cv.imwrite(f'calib_output/CalibratedPoints{i}.jpg', img)
        er_mat.append(er_sum1)
        
    return er_mat   




## this part is similar to the openCV documentation for cv.findChessboardCorners
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = 21.5*np.mgrid[1:10,1:7].T.reshape(-1,2)

world_corners = np.zeros([4,2])
world_corners = np.asarray(world_corners, dtype = np.float32)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
homographies = []
images = glob.glob('Calibration_Imgs/*.jpg')

start1 = timer()
impts = np.zeros_like(world_corners)
# Find image coordinates for set of world coordinates of checkerboard and find homography between 2
for i in (images):
    start2 = timer()
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6))
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners2)
        corners2 = np.asarray(corners2)
        corners2 = corners2.reshape((-1, 2))
        #cv.drawChessboardCorners(img, (9,6), corners2, ret)
        impts[0] = corners2[0]
        impts[1] = corners2[8]
        impts[2] = corners2[53]
        impts[3] = corners2[45]
        world_corners[0] = 21.5*np.float32([1, 1]) 
        world_corners[1] = 21.5*np.float32([9, 1]) 
        world_corners[2] = 21.5*np.float32([9, 6]) 
        world_corners[3] = 21.5*np.float32([1, 6])        
        H = cv.findHomography(world_corners, impts)
        homographies.append(H[0])
        cv.waitKey(10)


# find vector b and matrix V for equation Vb = 0 from the paper
V_final = []
for i in range(len(images)):
    h = homographies[i]
    h1 = h[:,0]
    h2 = h[:,1]
    h3 = h[:,2]
    
    v11 = [h[0][0]*h[0][0],
           (h[0][0]*h[1][0] + h[1][0]*h[0][0]),
           h[1][0]*h[1][0],
           (h[2][0]*h[0][0] + h[0][0]*h[2][0]),
           (h[2][0]*h[1][0] + h[1][0]*h[2][0]),
           h[2][0]*h[2][0]]
    v11 = np.asarray(v11)
    
    v22 = [h[0][1]*h[0][1],
           h[0][1]*h[1][1] + h[1][1]*h[0][1],
           h[1][1]*h[1][1],
           h[2][1]*h[0][1] + h[0][1]*h[2][1],
           h[2][1]*h[1][1] + h[1][1]*h[2][1],
           h[2][1]*h[2][1]]
    v22 = np.asarray(v22)

    v12 = [h[0][0]*h[0][1],
           (h[0][0]*h[1][1] + h[1][0]*h[0][1]),
           h[1][0]*h[1][1],
           (h[2][0]*h[0][1] + h[0][0]*h[2][1]),
           (h[2][0]*h[1][1] + h[1][0]*h[2][1]),
           h[2][0]*h[2][1]]
    v12 = np.asarray(v12)
    
    v_sec = v11 - v22
    V_final.append(v12)
    V_final.append(v_sec)

#Find V_t*V
V_final = np.asarray(V_final)
V_t = np.transpose(V_final)
V = np.dot(V_t,V_final)

#Find right singular eigen vector of V_final by doing SVD
U_svd, S, V_svd = np.linalg.svd(V_final)
b = (V_svd[-1,:])


#b = vector[min_eig_index]

B11 = b[0]
B12 = b[1]
B22 = b[2]
B13 = b[3]
B23 = b[4]
B33 = b[5]

# finding intrinsics of camera
v = (B12*B13 - B11*B23)/(B11*B22-B12**2)
lam = B33 - ((B13**2 + v*(B12*B13 - B11*B23))/B11)
alpha = np.sqrt(lam/B11)
beta = np.sqrt(lam*B11/(B11*B22-B12**2))
gamma = -B12*(alpha**2)*beta/lam
u = gamma*v/beta - B13*alpha**2/lam

intrinsic = np.zeros([3,3])
intrinsic[0][0] = alpha
intrinsic[0][1] = gamma
intrinsic[0][2] = u
intrinsic[1][1] = beta
intrinsic[1][2] = v
intrinsic[2][2] = 1

print('Initial estimation of intrinsic parameters:')
print(intrinsic)
#calculate extrinsics for each image
Extrinsic_mat = []
Rot_mat = []
lamb_mat = []
lamb1_mat = []
lamb2_mat = []
for i in range(len(images)):
    rot = np.zeros([3,3])
    ex_mat = np.zeros([3,4])
    h = homographies[i]
    h1 = h[:,0]
    h2 = h[:,1]
    h3 = h[:,2]
    A = intrinsic
    A_inv = np.linalg.inv(A)
    # find lambda for first column
    m1 = np.dot(A_inv,h1)
    s1 = np.sqrt(m1[0]**2 + m1[1]**2 + m1[2]**2)
    lamb1 = 1/s1
    
    # find lambda for second column
    m2 = np.dot(A_inv,h2)
    s2 = np.sqrt(m2[0]**2 + m2[1]**2 + m2[2]**2)
    lamb2 = 1/s2
    #averaging both lambda values as s1 and s2 have small difference on around 0.0001 which impacts the lambda value by 1. This is just for reference and is not used further in code
    lamb = (lamb1+lamb2)/2
    lamb_inv = np.linalg.norm(np.linalg.inv(A)@h1, ord=2)
    lamb = 1/lamb_inv
    r1 = lamb*(np.dot(A_inv,h1))
    r2 = lamb*(np.dot(A_inv,h2))
    r3 = np.cross(r1,r2)
    t = lamb*np.dot(A_inv,h3)
    #creating extrinsic matrix columns and appending the extrinsics for further use
    ex_mat[:,0] = r1
    ex_mat[:,1] = r2
    ex_mat[:,2] = r3
    ex_mat[:,3] = t
    Extrinsic_mat.append(ex_mat)
    lamb_mat.append(lamb)
    lamb1_mat.append(lamb1)
    lamb2_mat.append(lamb2)
    #storing the rotation matrices for reference
    rot = ex_mat[:,0:3]
    Rot_mat.append(rot)

K = np.transpose(np.zeros([1,2]))
k1 = K[0]
k2 = K[1]
#Finding total error for optimization part

# objpoints  - world points - M
# imgpoints - image pints - m
# m = H*M
# H = [r1 r2 t]
intrinsic_list = (alpha, gamma, u , beta, v, k1, k2)

k1 = 0
k2 = 0

initial_params = [alpha, gamma, u , beta, v, k1, k2]
err_init = error(initial_params, Extrinsic_mat, lamb_mat, objpoints, imgpoints)
print('Initial per point error:', np.mean(err_init))#/(13))
print('.........................................')
optimized_params = scipy.optimize.least_squares(error, initial_params, method = 'lm', args=[Extrinsic_mat, lamb_mat, objpoints, imgpoints])
values = optimized_params.x

intrinsic_final = np.zeros([3,3])
intrinsic_final[0][0] = values[0]
intrinsic_final[0][1] = values[1]
intrinsic_final[0][2] = values[2]
intrinsic_final[1][1] = values[3]
intrinsic_final[1][2] = values[4]
intrinsic_final[2][2] = 1
optimized_param_list = [values[0], values[1], values[2], values[3], values[4], values[5], values[6]]
err_optim = error(optimized_param_list, Extrinsic_mat, lamb_mat, objpoints, imgpoints)
print('Per point error after optimization:', np.mean(err_optim))#/(13))
print('Intrinsic parameters after optimization:')
print(intrinsic_final)
print('Lens distortion coefficients:')
print(values[5], values[6])
flag = 1
totalerror = error(optimized_param_list, Extrinsic_mat, lamb_mat, objpoints, imgpoints, flag)
