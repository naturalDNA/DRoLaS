from collections import defaultdict
import csv
import heapq
import random
import cv2 as cv
import pandas as pd
from skimage import morphology
import numpy as np
import os
from shapely.geometry import shape, Point, LineString
import geopandas as gpd
import matplotlib.pyplot as plt
#import openpyxl

# 如果路比这个短，那就定义为碎路，删除
minlength = 50
# 公制距离，随便定义
maxlength = 100
# 如果两个点之间的距离小于它，就合并为一个点
min_road = 5
# 如果两条路之间的夹角小于它，就合并为一条路
min_angle = 20


def distance(node1, node2):
    return (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2


def vector(node1, node2):
    return (node1[0] - node2[0], node1[1] - node2[1])


def cal_angle(vector1, vector2):
    arr_a = np.array(vector1)  # 向量a
    arr_b = np.array(vector2)  # 向量b
    cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
    if cos_value > 1:
        cos_value = 1.
    if cos_value < -1:
        cos_value = -1.
    return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度


def centerline_extraction(inraster):
    # print(image.dtype)
    img = inraster.copy()
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    #img = cv.dilate(img, kernel, iterations=5)
    img[img <= 200] = 0
    img[img > 200] = 1
    skeleton = morphology.skeletonize(img)
    skeleton = skeleton.astype(np.uint8) * 255
    return skeleton


def raster2vector(inraster):
    node_d = {"id": [], "geometry": [], "loction": []}
    edge_d = {"v1_id": [], "v2_id": [], 'geometry': []}
    node_dict = defaultdict(int)
    edge_dict = defaultdict(int)
    node_next_node = defaultdict(list)
    node_id = 0

    def random_walk(now_node):
        dist = {node: float('inf') for node in node_d["loction"]}
        heap = [(0, now_node)]
        visited = set()
        while heap:
            (d, current_node) = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            if d > dist[current_node]:
                continue
            for neighbor in node_next_node[current_node]:
                edge_weight = distance(neighbor, current_node) ** 0.5
                new_distance = d + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        ans = 0
        for x1, y1, x2, y2 in edge_dict.keys():
            node1, node2 = (x1, y1), (x2, y2)
            edge_length = distance(node1, node2) ** 0.5
            if dist[node1] >= maxlength and dist[node2] >= maxlength:
                continue
            elif max(0, maxlength - dist[node1]) + max(0, maxlength - dist[node2]) >= edge_length:
                ans += edge_length
            else:
                ans += max(0, maxlength - dist[node1]) + max(0, maxlength - dist[node2])
        return ans / 2

    def caculate_fluency(begin, end):
        dist = {node: float('inf') for node in node_d["loction"]}
        heap = [(0, begin)]
        visited = set()
        while heap:
            (d, current_node) = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            if d > dist[current_node]:
                continue
            for neighbor in node_next_node[current_node]:
                edge_weight = distance(neighbor, current_node) ** 0.5
                new_distance = d + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        # 返回起点到终点的最短距离
        return dist[end]

    def caculate_degree_num(degree):
        res = 0
        for node, keys in node_dict.items():
            if keys > 0:
                if len(node_next_node[node]) == degree:
                    res += 1
        return res

    def make_move(node, move, inraster):
        next_x, next_y = node[0] + move[0], node[1] + move[1]
        if 0 <= next_x <= 511 and 0 <= next_y <= 511:
            return inraster[next_x][next_y], (next_x, next_y)
        else:
            return 0, (next_x, next_y)

    def is_node(node, inraster):
        nodes = []
        moves = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for move in moves:
            action, next_node = make_move(node, move, inraster)
            if action:
                nodes.append(next_node)
        if len(nodes) == 2 and (nodes[0][0] + nodes[1][0]) // 2 == node[0] and (nodes[0][1] + nodes[1][1]) // 2 == node[
            1]:
            return False
        else:
            return True

    def find_node(inraster):
        nonlocal node_id
        w, h = inraster.shape
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                if inraster[x][y]:
                    if is_node([x, y], inraster):
                        node_dict[(x, y)] = node_id
                        node_id += 1

    def find_next_node(node, inraster, pre_node=None):
        nonlocal next_nodes
        moves = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for move in moves:
            action, next_node = make_move(node, move, inraster)
            if action:
                if next_node == pre_node:
                    continue
                elif node_dict[next_node]:
                    next_nodes.append(next_node)
                else:
                    find_next_node(next_node, inraster, node)

    find_node(inraster)
    node_list = list(node_dict.keys())
    # print("find all node")
    # 找到所有点的邻接顶点
    for node in node_list:
        next_nodes = []
        find_next_node(node, inraster)
        node_next_node[node] = next_nodes
    # print("find next_node")
    # 处理局部三角形
    for node in node_list:
        if node_dict[node]:
            if len(node_next_node[node]) > 2:
                for next_node in node_next_node[node]:
                    if node_dict[next_node]:
                        flag = 0
                        for next_next_node in node_next_node[next_node]:
                            if node_dict[next_next_node]:
                                if next_next_node in node_next_node[node]:
                                    flag = 1
                                    mid_node = next_next_node
                                    break
                        if flag:
                            if distance(node, mid_node) == 2:
                                try:
                                    node_next_node[node].remove(mid_node)
                                except Exception as e:
                                    pass
                                try:
                                    node_next_node[mid_node].remove(node)
                                except Exception as e:
                                    pass

                            if distance(node, next_node) == 2:
                                try:
                                    node_next_node[node].remove(next_node)
                                except Exception as e:
                                    pass          
                                try:                                                          
                                    node_next_node[next_node].remove(node)
                                except Exception as e:
                                    pass  
                            if distance(next_node, mid_node) == 2:
                                try:
                                    node_next_node[next_node].remove(mid_node)
                                except Exception as e:
                                    pass          
                                try:                                
                                    node_next_node[mid_node].remove(next_node)
                                except Exception as e:
                                    pass    

    # print("处理局部三角形")
    # 合并度为2的点
    for i in range(10):
        for node in node_list:
            if node_dict[node]:
                if len(node_next_node[node]) == 2:
                    first_node, next_node = node_next_node[node]
                    vector1 = vector(first_node, node)
                    vector2 = vector(node, next_node)
                    angle = cal_angle(vector1, vector2)
                    dis = min(distance(node, first_node) ** 0.5, distance(node, next_node) ** 0.5)
                    if 0 <= angle <= min_angle or dis <= min_road:
                        node_dict[node] = 0
                        node_next_node[first_node].remove(node)
                        node_next_node[first_node].append(next_node)
                        node_next_node[next_node].remove(node)
                        node_next_node[next_node].append(first_node)
    # print("合并度为2的点")

    # smooth ，但是感觉效果一般，还是算了
    # for node in node_list:
    #     if node_dict[node]:
    #         if len(node_next_node[node])!=2:
    #             for next_node in node_next_node[node]:
    #                 s=[node]
    #                 while True:
    #                     s.append(next_node)
    #                     if len(node_next_node[next_node])==2:
    #                         next_node = node_next_node[next_node][1] if node_next_node[next_node][0]==s[-2] else node_next_node[next_node][0]
    #                     else:
    #                         break

    #                 node_nums=len(s)
    #                 for i in range(1,node_nums-1):
    #                     node1,node2,node3=s[i],s[i-1],s[i+1]
    #                     new_node=((node2[0]//4+node3[0]//4+node1[0]//2),(node2[1]//4+node3[1]//4+node1[1]//2))
    #                     if node_dict[new_node]==0:
    #                         node_dict[new_node]=node_dict[node1]
    #                         node_dict[node1]=0
    #                         node_next_node[node2].remove(node1)
    #                         node_next_node[node2].append(new_node)
    #                         node_next_node[node3].remove(node1)
    #                         node_next_node[node3].append(new_node)
    #                         node_next_node[new_node]=[node2,node3]
    #                         s[i]=new_node
    # print("smooth")
    # node_list = list(node_dict.keys())
    # 删除碎路
    for node in node_list:
        if node_dict[node]:
            queue = [node]
            candidate = [node]
            total_length = 0
            while queue:
                now_node = queue.pop()
                for next_node in node_next_node[now_node]:
                    if node_dict[next_node] and next_node not in candidate:
                        queue.append(next_node)
                        candidate.append(next_node)
                        total_length += distance(now_node, next_node) ** 0.5
            if total_length <= minlength:
                for remove_node in candidate:
                    node_dict[remove_node] = 0
    # print("删除碎路")
    # 根据每个点的邻接顶点构造边
    total_length = 0
    for node in node_list:
        id = node_dict[node]
        if id:
            node_d["id"].append(id)
            node_d["loction"].append(node)
            node_d["geometry"].append(Point(node[0], node[1]))
            for next_node in node_next_node[node]:
                x, y, next_x, next_y = node[0], node[1], next_node[0], next_node[1]
                if edge_dict[(x, y, next_x, next_y)] == 0:
                    edge_dict[(x, y, next_x, next_y)] = 1
                    edge_dict[(next_x, next_y, x, y)] = 1
                    now_id = node_dict[(x, y)]
                    next_id = node_dict[(next_x, next_y)]
                    prev_node = Point(x, y)
                    node_next = Point(next_x, next_y)
                    edge = LineString([prev_node, node_next])
                    total_length += distance(node, next_node) ** 0.5
                    edge_d["v1_id"].append(now_id)
                    edge_d["v2_id"].append(next_id)
                    edge_d["geometry"].append((edge))
    # print("构造边")

    total_node_num = len(node_d["id"])
    total_edge_num = len(edge_d["v2_id"])
    total = 0
    for _ in range(100):
        try:
            v1 = random.choice(node_d["loction"])
            v2 = random.choice(node_d["loction"])
            kkk=0
            while distance(v1, v2) ** 0.5 < 50:
                v1 = random.choice(node_d["loction"])
                v2 = random.choice(node_d["loction"])
                kkk+=1
                if kkk>50:
                    break
            dist = caculate_fluency(v1, v2)
            kkk=0
            while dist == float("inf") or abs(dist) < 1e-6:
                v1 = random.choice(node_d["loction"])
                v2 = random.choice(node_d["loction"])
                dist = caculate_fluency(v1, v2)
                kkk+=1
                if kkk>50:
                    break
            o_dist = distance(v1, v2) ** 0.5
            total += o_dist / dist
        except Exception as e:
            pass
        # print("第{}对,v1{},v2{},欧氏距离{},最短路径{}".format(i,v1,v2,o_dist,dist))
    total2 = 0
    # for _ in range(100):
    #     v = random.choice(node_d["loction"])
    #     walk_length = random_walk(v)
    #     # print("第{}对,v{},公制距离{}".format(_,v,walk_length))
    #     total2 += walk_length
    if total_node_num==0:
        conectivity=1
    else:
        conectivity=total_edge_num * 2 / total_node_num
    if total_edge_num==0:        
        aver_road_length=0
    else:
        aver_road_length=total_length * 5 / total_edge_num

    # print("***********************************************")
    # print("总结点数为{}".format(total_node_num))
    # print("街道数为{}".format(total_edge_num))
    # print("连通性为{}".format(total_edge_num/total_node_num))
    # print("街道总长度为{}米".format(total_length*5))
    # print("平均街道长度为{}米".format(total_length*5/total_edge_num))
    # print("平均交流便利性为{}".format(total/25))
    # print("平均公制范围为{}米".format(total2/25))
    # print("度为1的点有{}".format(caculate_degree_num(1)))
    # print("度为2的点有{}".format(caculate_degree_num(2)))
    # print("度为3的点有{}".format(caculate_degree_num(3)))
    # print("度为4的点有{}".format(caculate_degree_num(4)))
    # print("***********************************************")
    #print(total_node_num)
    return node_d, edge_d, [total_node_num, total_edge_num, conectivity, total_length * 5,
                            aver_road_length,
                            total / 100, total2 / 100, caculate_degree_num(1), caculate_degree_num(2),
                            caculate_degree_num(3), caculate_degree_num(4)]

def raster2vector_norefine(inraster):
    node_d = {"id": [], "geometry": [], "loction": []}
    edge_d = {"v1_id": [], "v2_id": [], 'geometry': []}
    node_dict = defaultdict(int)
    edge_dict = defaultdict(int)
    node_next_node = defaultdict(list)
    node_id = 0

    def random_walk(now_node):
        dist = {node: float('inf') for node in node_d["loction"]}
        heap = [(0, now_node)]
        visited = set()
        while heap:
            (d, current_node) = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            if d > dist[current_node]:
                continue
            for neighbor in node_next_node[current_node]:
                edge_weight = distance(neighbor, current_node) ** 0.5
                new_distance = d + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        ans = 0
        for x1, y1, x2, y2 in edge_dict.keys():
            node1, node2 = (x1, y1), (x2, y2)
            edge_length = distance(node1, node2) ** 0.5
            if dist[node1] >= maxlength and dist[node2] >= maxlength:
                continue
            elif max(0, maxlength - dist[node1]) + max(0, maxlength - dist[node2]) >= edge_length:
                ans += edge_length
            else:
                ans += max(0, maxlength - dist[node1]) + max(0, maxlength - dist[node2])
        return ans / 2

    def caculate_fluency(begin, end):
        dist = {node: float('inf') for node in node_d["loction"]}
        heap = [(0, begin)]
        visited = set()
        while heap:
            (d, current_node) = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            if d > dist[current_node]:
                continue
            for neighbor in node_next_node[current_node]:
                edge_weight = distance(neighbor, current_node) ** 0.5
                new_distance = d + edge_weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        # 返回起点到终点的最短距离
        return dist[end]

    def caculate_degree_num(degree):
        res = 0
        for node, keys in node_dict.items():
            if keys > 0:
                if len(node_next_node[node]) == degree:
                    res += 1
        return res

    def make_move(node, move, inraster):
        next_x, next_y = node[0] + move[0], node[1] + move[1]
        if 0 <= next_x <= 511 and 0 <= next_y <= 511:
            return inraster[next_x][next_y], (next_x, next_y)
        else:
            return 0, (next_x, next_y)

    def is_node(node, inraster):
        nodes = []
        moves = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for move in moves:
            action, next_node = make_move(node, move, inraster)
            if action:
                nodes.append(next_node)
        if len(nodes) == 2 and (nodes[0][0] + nodes[1][0]) // 2 == node[0] and (nodes[0][1] + nodes[1][1]) // 2 == node[
            1]:
            return False
        else:
            return True

    def find_node(inraster):
        nonlocal node_id
        w, h = inraster.shape
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                if inraster[x][y]:
                    if is_node([x, y], inraster):
                        node_dict[(x, y)] = node_id
                        node_id += 1

    def find_next_node(node, inraster, pre_node=None):
        nonlocal next_nodes
        moves = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for move in moves:
            action, next_node = make_move(node, move, inraster)
            if action:
                if next_node == pre_node:
                    continue
                elif node_dict[next_node]:
                    next_nodes.append(next_node)
                else:
                    find_next_node(next_node, inraster, node)

    find_node(inraster)
    node_list = list(node_dict.keys())
    #print("find all node")
    # 找到所有点的邻接顶点
    for node in node_list:
        next_nodes = []
        try: 
            find_next_node(node, inraster)
        except Exception as e:
            pass          
        node_next_node[node] = next_nodes
    #print("find next_node")
    # 处理局部三角形
    for node in node_list:
        if node_dict[node]:
            if len(node_next_node[node]) > 2:
                for next_node in node_next_node[node]:
                    if node_dict[next_node]:
                        flag = 0
                        for next_next_node in node_next_node[next_node]:
                            if node_dict[next_next_node]:
                                if next_next_node in node_next_node[node]:
                                    flag = 1
                                    mid_node = next_next_node
                                    break
                        if flag:
                            if distance(node, mid_node) == 2:
                                try:
                                    node_next_node[node].remove(mid_node)
                                except Exception as e:
                                    pass
                                try:
                                    node_next_node[mid_node].remove(node)
                                except Exception as e:
                                    pass

                            if distance(node, next_node) == 2:
                                try:
                                    node_next_node[node].remove(next_node)
                                except Exception as e:
                                    pass          
                                try:                                                          
                                    node_next_node[next_node].remove(node)
                                except Exception as e:
                                    pass  
                            if distance(next_node, mid_node) == 2:
                                try:
                                    node_next_node[next_node].remove(mid_node)
                                except Exception as e:
                                    pass          
                                try:                                
                                    node_next_node[mid_node].remove(next_node)
                                except Exception as e:
                                    pass    

    #print("处理局部三角形")
    # 合并度为2的点
    for i in range(10):
        for node in node_list:
            if node_dict[node]:
                if len(node_next_node[node]) == 2:
                    first_node, next_node = node_next_node[node]
                    vector1 = vector(first_node, node)
                    vector2 = vector(node, next_node)
                    angle = cal_angle(vector1, vector2)
                    dis = min(distance(node, first_node) ** 0.5, distance(node, next_node) ** 0.5)
                    if 0 <= angle <= min_angle or dis <= min_road:
                        node_dict[node] = 0
                        try: 
                            node_next_node[first_node].remove(node)
                            node_next_node[first_node].append(next_node)
                            node_next_node[next_node].remove(node)
                            node_next_node[next_node].append(first_node)
                        except Exception as e:
                            pass    
    #print("合并度为2的点")
    # 删除碎路
    
    # 根据每个点的邻接顶点构造边
    total_length = 0
    for node in node_list:
        id = node_dict[node]
        if id:
            node_d["id"].append(id)
            node_d["loction"].append(node)
            node_d["geometry"].append(Point(node[0], node[1]))
            for next_node in node_next_node[node]:
                x, y, next_x, next_y = node[0], node[1], next_node[0], next_node[1]
                if edge_dict[(x, y, next_x, next_y)] == 0:
                    edge_dict[(x, y, next_x, next_y)] = 1
                    edge_dict[(next_x, next_y, x, y)] = 1
                    now_id = node_dict[(x, y)]
                    next_id = node_dict[(next_x, next_y)]
                    prev_node = Point(x, y)
                    node_next = Point(next_x, next_y)
                    edge = LineString([prev_node, node_next])
                    total_length += distance(node, next_node) ** 0.5
                    edge_d["v1_id"].append(now_id)
                    edge_d["v2_id"].append(next_id)
                    edge_d["geometry"].append((edge))
    #print("构造边")

    total_node_num = len(node_d["id"])
    total_edge_num = len(edge_d["v2_id"])
    total = 0

    for _ in range(100):
        try:
            v1 = random.choice(node_d["loction"])
            v2 = random.choice(node_d["loction"])
            kkk=0
            while distance(v1, v2) ** 0.5 < 50:
                v1 = random.choice(node_d["loction"])
                v2 = random.choice(node_d["loction"])
                kkk+=1
                if kkk>50:
                    break
            dist = caculate_fluency(v1, v2)
            kkk=0
            while dist == float("inf") or abs(dist) < 1e-6:
                v1 = random.choice(node_d["loction"])
                v2 = random.choice(node_d["loction"])
                dist = caculate_fluency(v1, v2)
                kkk+=1
                if kkk>50:
                    break
            o_dist = distance(v1, v2) ** 0.5
            total += o_dist / dist
        except Exception as e:
            pass
        #print("第{}对,v1{},v2{},欧氏距离{},最短路径{}".format(i,v1,v2,o_dist,dist))
    total2 = 0
    # for _ in range(100):
    #     v = random.choice(node_d["loction"])
    #     walk_length = random_walk(v)
    #     # print("第{}对,v{},公制距离{}".format(_,v,walk_length))
    #     total2 += walk_length
    if total_node_num==0:
        conectivity=1
    else:
        conectivity=total_edge_num * 2 / total_node_num
    if total_edge_num==0:        
        aver_road_length=0
    else:
        aver_road_length=total_length * 5 / total_edge_num

    return node_d, edge_d, [total_node_num, total_edge_num, conectivity, total_length * 5,
                            aver_road_length,
                            total / 100, total2 / 100, caculate_degree_num(1), caculate_degree_num(2),
                            caculate_degree_num(3), caculate_degree_num(4)]

def get_conn(net):
    dic = {"id": [], "nodes_num": [], "edges_num": [], "road_length": [], "conectivity": [], "aver_road_length": [],
           "frequency": [],
           "metric_range": [], "degree_1": [], "degree_2": [], "degree_3": [], "degree_4": []}
    inraster = cv.imread(net,2)
    inraster = centerline_extraction(inraster)
    nodes, edges, eval = raster2vector(inraster)
    dic["id"].append('1')
    dic["nodes_num"].append(eval[0])
    dic["edges_num"].append(eval[1])
    dic["conectivity"].append(eval[2])
    dic["road_length"].append(eval[3])
    dic["aver_road_length"].append(eval[4])
    dic["frequency"].append(eval[5])
    dic["metric_range"].append(eval[6])
    dic["degree_1"].append(eval[7])
    dic["degree_2"].append(eval[8])
    dic["degree_3"].append(eval[9])
    dic["degree_4"].append(eval[10])
    #out_nodes = gpd.GeoDataFrame(nodes)
    #out_edges = gpd.GeoDataFrame(edges)
    #print(dic["road_length"])
    #out_nodes.to_file("nodes/{}.shp".format('1'), driver='ESRI Shapefile')
    #out_edges.to_file("edges/{}.shp".format('1'), driver='ESRI Shapefile')
    #print("successful saves {} {}".format('1', '1'))
    return dic["conectivity"][0]

