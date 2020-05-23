# -*- coding: utf-8 -*-
import numpy as np
import cv2

#스크린 변수(단위: 픽셀)
w = 800 #스크린 너비
h = 600 #스크린 폭
f = 100 #초점거리

# 카메라 좌표계 -> OpenCV 좌표계 변환 함수
def opencv_coordinate(point=(0,0,0,1)):
    if point==None: return None
    x_cam, _, z_cam, _ = point
    #if abs(x_cam)>w/2+1 or abs(z_cam)>h/2+1: return None
    x_opencv = int(round(w/2 + x_cam))
    y_opencv = int(round(h/2 - z_cam))
    return (x_opencv, y_opencv)

#색상 정의
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
yellow_color = tuple((np.array(red_color) + np.array(green_color)).tolist())
white_color = (255, 255, 255)

#기준 좌표계의 원점 좌표축 정의
origin = (0,0,0,1)
axes = [origin]+[(1000,0,0,1), (0,1000,0,1), (0,0,1000,1)]
axes_edge_connected = ((1,1,1,1),
                       (1,0,0,0),
                       (1,0,0,0),
                       (1,0,0,0))

#카메라의 위치와 방향을 결정하는 변수
a = 400 #카메라 x좌표(기준 좌표계 기준)
b = -3000 #카메라 y좌표
c = 400 #카메라 z좌표
theta_x = 0
theta_y = 0
theta_z = np.pi/6

#그릴 도형 정의
polyhedron = [(-1000,-1000,1000,1), (1000,-1000,1000,1), (1000,1000,1000,1), (-1000,1000,1000,1),
        (-1000,-1000,-1000,1), (1000,-1000,-1000,1), (1000,1000,-1000,1), (-1000,1000,-1000,1)]
polyhedron_edge_connected = ((1,1,0,1,1,0,0,0),
                       (1,1,1,0,0,1,0,0),
                       (0,1,1,1,0,0,1,0),
                       (1,0,1,1,0,0,0,1),
                       (1,0,0,0,1,1,0,1),
                       (0,1,0,0,1,1,1,0),
                       (0,0,1,0,0,1,1,1),
                       (0,0,0,1,1,0,1,1))

# 좌표계 변환 T_coord 정의
def T_coord():
    T_rotation = np.array([[np.cos(theta_y)*np.cos(theta_z), -np.cos(theta_y)*np.sin(theta_z), np.sin(theta_y), 0],
                            [np.cos(theta_x)*np.sin(theta_z)+np.cos(theta_z)*np.sin(theta_x)*np.sin(theta_y), np.cos(theta_x)*np.cos(theta_z)-np.sin(theta_x)*np.sin(theta_y)*np.sin(theta_z), -np.sin(theta_x)*np.cos(theta_y), 0],
                            [np.sin(theta_x)*np.sin(theta_z)-np.cos(theta_x)*np.cos(theta_z)*np.sin(theta_y), np.cos(theta_z)*np.sin(theta_x)+np.cos(theta_x)*np.sin(theta_y)*np.sin(theta_z),  np.cos(theta_x)*np.cos(theta_y), 0],
                            [0,0,0,1]])
    T_parallel = np.array([[1,0,0,-a],
                           [0,1,0,-b],
                           [0,0,1,-c],
                           [0,0,0,1]])
    T = T_rotation.T.dot(T_parallel)
    return T

# 원근투영 변환 T_proj 정의
def T_proj(point=(1,1,1,1)):
    x, y, z, _ = point
    if y>0: proj_point = (x*f/y, f, z*f/y, 1)
    else: proj_point = None
    return opencv_coordinate(proj_point)

# 기준 좌표계 위의 점을 카메라 좌표계의 스크린에 원근투영하는 변환 T_composition
def T_composition(point=(1,1,1,1)):
    proj_point = T_proj(T_coord().dot(point))
    return proj_point


#모서리 처리 
def edge_proj(polyhedron, polyhedron_edge_connected):
    edges = []
    new_edges = []
    
    for i, start_coord in enumerate(polyhedron):
        for j, end_coord in enumerate(polyhedron):
            if i<j and polyhedron_edge_connected[i][j]:
                #edges.append([start_coord, end_coord] )
                edges.append([T_coord().dot(start_coord).tolist()[:3], T_coord().dot(end_coord).tolist()[:3]] )

    for edge in edges:
        start_coord, end_coord = np.array(edge[0]), np.array(edge[1])
        difference = end_coord - start_coord
        magnitude = np.sqrt(difference.dot(difference))
        num_of_points = int(magnitude)+1
        new_edges.append( [T_proj((start_coord+difference/num_of_points*t).tolist()+[1]) for t in range(1,num_of_points)] )
    return new_edges # [[[x1,y1],[x2,y2]], ...]


#스크린에 그리기
for t in range(15):
    #화면 초기화
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    #기준 좌표계의 원점과 축 그리기        
    axes_proj = edge_proj(axes, axes_edge_connected)
    for axis, color in zip(axes_proj, [red_color, green_color, blue_color]):
        for i, start_coord in enumerate(axis[:-2]):
            if start_coord!=None and axis[i+1]!=None:
                cv2.line(img, start_coord, axis[i+1], color, 3)
        cv2.arrowedLine(img, axis[i+1], axis[i+2], color, 10)
    
    #입체도형의 꼭짓점 그리기
    qs = [T_composition(p_i) for p_i in polyhedron]
    
    for qi in qs: cv2.line(img, qi, qi, yellow_color, 10)
    
    #입체도형의 모서리 그리기
    edges_proj = edge_proj(polyhedron, polyhedron_edge_connected)
    for edge in edges_proj:
        for i, start_coord in enumerate(edge[:-1]):
            if start_coord!=None and edge[i+1]!=None:
                cv2.line(img, start_coord, edge[i+1], white_color, 3)
    
    cv2.imshow('image', img)
    
    cv2.waitKey(0)
    b+=200 #시간에 따라 카메라 y좌표 변화
    a-=50
    theta_y -= 0.03*np.pi

cv2.destroyAllWindows()
