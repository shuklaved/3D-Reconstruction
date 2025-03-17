import cv2
import numpy as np

def Reprojection3D(image, disparity, f, b):
    Q = np.array([[1, 0, 0, -1307.728], [0, 1, 0, -1011.728], [0, 0, 0, f], [0, 0, -1 / b, -500 / b]])

    points = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > disparity.min()
    colors = image

    out_points = points[mask]
    out_colors = image[mask]

    verts = out_points.reshape(-1, 3)
    colors = out_colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    with open('/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/bin_3d.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

b = 193.001
f = 3997.684

imgL = cv2.imread("/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/im0.png")
disparity = cv2.imread("/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/disp.png", cv2.IMREAD_GRAYSCALE)
disparity = np.int16(disparity)
Reprojection3D(imgL, disparity, f, b)
