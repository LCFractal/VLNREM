# coding=utf-8

from __future__ import print_function

import argparse
import collections
import json
import pickle
import ast
import os
import random

import graph_utils

import networkx as nx
import numpy as np

PROPORTION = 0.5
random.seed(0)

def select_viewpoint(paths):
    viewpoint={}
    def _add_viewpoint(path):
        if path not in viewpoint.keys():
            viewpoint[path] = 1
        else:
            viewpoint[path] += 1
    for path in paths:
        for chunks in path['chunk_view']:
            for i in range(len(chunks)-1):
                if chunks[i][1]==chunks[i+1][0]:
                    _add_viewpoint(path['path'][chunks[i][1]-1])
    return viewpoint

def select_edge(paths,ebcnode,shortest_distance,pos2d):
    edge = {}
    for key in ebcnode.keys():
        x, y = tuple(pos2d[key[1]] - pos2d[key[0]])
        heading1 = np.arctan2( y,  x) % (2 * np.pi)
        heading2 = np.arctan2(-y, -x) % (2 * np.pi)
        vi1 = int(round(heading1 * 6 / np.pi) % 12)
        vi2 = int(round(heading2 * 6 / np.pi) % 12)
        edge[key] = {'count':0,
                     'conflict':set(),
                     'edge':key,
                     'edgepos2d':[pos2d[key[0]],pos2d[key[1]]],
                     'edgevi': [vi1,vi2],
                     's_e':{'s':{},'m':{}},
                     'e_s':{'s':{},'m':{}}}
    def _add_edge(path,start,end,sub_path,sub_instruction,s2e=1):

        sub_distance = sum([shortest_distance[sub_path[i]][sub_path[i+1]] for i in range(len(sub_path)-1)])
        sub_distance = round(sub_distance,2)
        sub_scan = path['scan']
        sub_path_id = path['path_id']
        sub_heading = path['heading']
        sub_start = s2e

        tmp = ''
        for sub_str in sub_instruction:
            for sub_sub_str in sub_str:
                tmp = tmp+' '+sub_sub_str
        sub_instruction = tmp[1:]

        start = path['path'][start]
        end = path['path'][end]
        s_e = (start,end)
        se_or_es = 's_e'
        if s_e not in edge.keys():
            s_e = (end,start)
            se_or_es = 'e_s'

        if sub_start == 1:
            s_or_m='s'
        else:
            s_or_m='m'
            x, y = pos2d[sub_path[0]] - pos2d[sub_path[1]]
            sub_heading = np.arctan2(y, x) % (2 * np.pi)
            sub_heading = round(sub_heading, 3)

        for val in sub_path:
            edge[s_e]['conflict'].add(val)

        edge[s_e]['count'] += 1
        if sub_path_id not in edge[s_e][se_or_es][s_or_m].keys():
            edge[s_e][se_or_es][s_or_m][sub_path_id]=\
            {
                'distance': sub_distance,
                'scan': sub_scan,
                'path_id': sub_path_id,
                'path': sub_path,
                'heading': sub_heading,
                'instructions': [sub_instruction],
                's_or_m': s_or_m,
                'se_or_es': se_or_es,
            }
        else:
            edge[s_e][se_or_es][s_or_m][sub_path_id]['instructions'].append(sub_instruction)

    for path in paths:
        for index, chunks in enumerate(path['chunk_view']):
            for i in range(len(chunks)-1):
                if chunks[i][1]==chunks[i+1][0]:
                    if chunks[i][0] != chunks[i][1]:
                        sub_path = path['path'][:chunks[i][1]]
                        new_instruction = path['new_instructions'][index][:(i+1)]
                        _add_edge(path,chunks[i][1]-2,chunks[i][1]-1,
                                  sub_path,new_instruction,s2e=1)
                    if chunks[i + 1][0] != chunks[i + 1][1]:
                        sub_path = path['path'][chunks[i][1]-1:]
                        sub_instruction = path['new_instructions'][index][(i + 1):]
                        _add_edge(path,chunks[i][1]-1,chunks[i][1],
                                  sub_path,sub_instruction,s2e=0)
    return edge

def main(args):

  print('******Visualization R2R Graph********')

  def _connections_file_path(scan):
    return os.path.join(
        args.connections_dir, '{}_connectivity.json'.format(scan))

  def _take_second(elem):
      return elem[2]

  inputs = json.load(open(args.input_file_path))

  # Group by scan to save memory.
  scans = dict()
  for value in inputs:
    scan = value['scan']
    value['new_instructions'] = ast.literal_eval(value['new_instructions'])
    if scan not in scans:
      scans[scan] = []
    scans[scan].append(value)

  sub_scans = dict()
  for scan, values in scans.items():
    print('Loading graph for scan {}.'.format(scan))
    graph = graph_utils.load(_connections_file_path(scan))
    pos2d = nx.get_node_attributes(graph, 'pos2d')

    path_distance_list = [v['distance'] for v in values]
    path_distance_max = max(path_distance_list)

    cache = dict(nx.all_pairs_dijkstra(graph, weight='weight3d'))
    shortest_distance = {k: v[0] for k, v in cache.items()}
    shortest_path = {k: v[1] for k, v in cache.items()}

    # cache = list(nx.all_node_cuts(graph))
    # cutnode = [ str(list(n)[0]) for n in cache]
    bcnode = nx.betweenness_centrality(graph)
    viewpoint = select_viewpoint(values)
    cache = [[k,v,bcnode[k]] for k, v in viewpoint.items()]
    cache.sort(key=_take_second, reverse=True)
    cache = np.array(cache)
    viewnode = [v[0] for v in cache[0:30]]

    ebcnode = nx.edge_betweenness_centrality(graph)
    all_edge = select_edge(values,ebcnode,shortest_distance,pos2d)
    cache = [[k,ebcnode[k], v['count'],v] for k, v in all_edge.items() if v != 0]
    cache.sort(key=_take_second, reverse=True)
    # cache = np.array(cache)
    edge = [v[0] for v in cache[0:30]]

    # cache = [[k, v] for k, v in nx.betweenness_centrality(graph).items()]
    # cache.sort(key=_take_second,reverse=True)
    # bcnode = [v[0] for v in cache[0:20]]

    # graph_utils.draw(graph,viewnode,edge,'Graph/'+scan+'.png')
    cache = graph_utils.draw_edge(graph,viewnode,edge,'Graph/'+scan+'.png')
    # graph_utils.draw(graph,values[0]['path'],values[1]['path'],'Graph/'+scan+'.png')
    # Cache format: (node, (distance, path)) ((node obj, (dict, dict)))

    common_edges = dict()
    for s_e in list(cache):
        if s_e in all_edge.keys():
            common_edges[s_e] = all_edge[s_e]
        else:
            s_e=(s_e[1],s_e[0])
            common_edges[s_e] = all_edge[s_e]

    max_counts = [0,0,0]
    for first_edge in common_edges:
        conflict = common_edges[first_edge]['conflict']
        count = common_edges[first_edge]['count']
        max_count = [0,0,0]

        for second_edge in common_edges:
            if first_edge==second_edge:
                continue
            v1,v2 = second_edge
            if (v1 in conflict) or (v2 in conflict):
                continue
            count2 = common_edges[second_edge]['count']
            if count+count2>max_count[0]:
                max_count = [count+count2,first_edge,second_edge]
        if max_count[0] == 0:
            max_count=[count,first_edge,0]
        common_edges[first_edge]['max_count'] = max_count


        if max_count[0]>max_counts[0]:
            max_counts = max_count

    sub_scans[scan] = [max_counts,common_edges]

  with open(args.mid_file_path, 'wb') as f:
      pickle.dump(sub_scans, f)
  # Dataset summary metrics.
  num_graph = len(sub_scans)
  tot_subinstructions = np.sum([sub_scans[scan][0][0] for scan in sub_scans])
  avg_subinstructions = np.mean([sub_scans[scan][0][0] for scan in sub_scans])

  print('******Final Results********')
  print('  Number of graph:             {}'.format(num_graph))
  print('  Total sub instructions:      {}'.format(tot_subinstructions))
  print('  Average sub instructions:    {}'.format(avg_subinstructions))

def _combine_instructions(ins1,ins2):
    com_ins = []
    for in1 in ins1:
        for in2 in ins2:
            com_ins.append(in1+', '+in2+'.')
    random.shuffle(com_ins)
    return com_ins[:3]

def generate_RER2R(args):

    print('******Generate R2R Graph********')

    def _connections_file_path(scan):
        return os.path.join(
            args.connections_dir, '{}_connectivity.json'.format(scan))

    sub_scans = pickle.load(open(args.mid_file_path,'rb'))

    tmp = dict()
    for key in sub_scans:
        count = sub_scans[key][0]
        edge = sub_scans[key][1]
        if count[2] == 0:
            continue
        first_edge = edge[count[1]]
        second_edge = edge[count[2]]
        tmp[key] = {
            'count':count[0],
            'edge': (count[1],count[2]),
            'first_edge':first_edge,
            'second_edge':second_edge
        }
    sub_scans = tmp

    key1 = list(sub_scans.keys())
    key2 = list(sub_scans.keys())
    random.shuffle(key1)
    random.shuffle(key2)

    sub_graph = {}
    new_paths = []

    def _combine_path(path_set1,path_set2,edge1,edge2,pos2d1,pos2d2,scan1,scan2,vi1,vi2):
        global len_ins
        x, y = pos2d1[1] - pos2d1[0]
        p_heading1 = round(np.arctan2(y, x) % (2 * np.pi),3)
        x, y = pos2d2[1] - pos2d2[0]
        p_heading2 = round(np.arctan2(y, x) % (2 * np.pi),3)
        x, y = pos2d1[0] - pos2d1[1]
        se_heading1 = round(np.arctan2(y, x) % (2 * np.pi),3)
        x, y = pos2d2[0] - pos2d2[1]
        se_heading2 = round(np.arctan2(y, x) % (2 * np.pi),3)

        x, y = pos2d2[0] - pos2d1[0]
        rl_heading1 = round(np.arctan2(y, x) % (2 * np.pi), 3)
        vri1 = int(round(rl_heading1 * 6 / np.pi) % 12)

        x, y = pos2d2[0] - pos2d1[0]
        rl_heading1 = round(np.arctan2(y, x) % (2 * np.pi), 3)
        vri2 = int(round(rl_heading1 * 6 / np.pi) % 12)

        shortest = sub_graph[scan1+'+'+scan2]['shortest_distance']

        for p1 in path_set1:
            p1 = path_set1[p1]
            for p2 in path_set2:
                p2 = path_set2[p2]
                path = p1['path'][:-1]+p2['path'][1:]
                if path[-1] in list(shortest[path[0]].keys()):
                    distance = round(shortest[path[0]][path[-1]],2)
                else:
                    continue
                new_item = {
                    'distance': distance,
                    'scan': scan1+'+'+scan2,
                    'path_id': int(p1['path_id']*1e9+p2['path_id']),
                    'path': path,
                    'heading': p1['heading'],
                    'instructions': _combine_instructions(
                        p1['instructions'],p2['instructions']),
                    'second': {
                        'scan1': scan1,
                        'scan2': scan2,
                        'heading1': p_heading1,
                        'heading2': p_heading2,
                        'edge1': edge1,
                        'edge2': edge2,
                        'se_heading1': se_heading1,
                        'se_heading2': se_heading2,
                        'vi1': vi1,
                        'vi2': vi2,
                        'vri1': vri1,
                        'vri2': vri2
                    }
                }
                new_paths.append(new_item)

    for index in range(len(key1)):
        print("Generating: {}".format(index))
        scan1 = key1[index]
        scan2 = key2[index]
        if scan1 == scan2:
            continue
        sub_scan1 = sub_scans[scan1]
        sub_scan2 = sub_scans[scan2]

        graph_utils.save(args.connections_dir, _connections_file_path(scan1), _connections_file_path(scan2), scan1,
                         scan2, sub_scan1, sub_scan2)
        graph_utils.save(args.connections_dir, _connections_file_path(scan2), _connections_file_path(scan1), scan2,
                         scan1, sub_scan2, sub_scan1)

        for key in [scan1+'+'+scan2, scan2+'+'+scan1]:
            tmpgraph = graph_utils.load(_connections_file_path(key))
            cache = dict(nx.all_pairs_dijkstra(tmpgraph, weight='weight3d'))
            tmpshortest = {k: v[0] for k, v in cache.items()}
            sub_graph[key] = {
                'graph': tmpgraph,
                'shortest_distance': tmpshortest
            }

        for i_edge,x_edge in enumerate(['first_edge','second_edge']):
            # scan1 to scan2
            _combine_path(sub_scan1[x_edge]['s_e']['s'],
                          sub_scan2[x_edge]['s_e']['m'],
                          sub_scan1['edge'][i_edge],
                          sub_scan2['edge'][i_edge],
                          sub_scan1[x_edge]['edgepos2d'],
                          [sub_scan2[x_edge]['edgepos2d'][1], sub_scan2[x_edge]['edgepos2d'][0]],
                          scan1, scan2,
                          sub_scan1[x_edge]['edgevi'],
                          sub_scan2[x_edge]['edgevi'])

            _combine_path(sub_scan1[x_edge]['e_s']['s'],
                          sub_scan2[x_edge]['e_s']['m'],
                          (sub_scan1['edge'][i_edge][1],sub_scan1['edge'][i_edge][0]),
                          (sub_scan2['edge'][i_edge][1],sub_scan2['edge'][i_edge][0]),
                          [sub_scan1[x_edge]['edgepos2d'][1],sub_scan1[x_edge]['edgepos2d'][0]],
                          sub_scan2[x_edge]['edgepos2d'],
                          scan1, scan2,
                          [sub_scan1[x_edge]['edgevi'][1],sub_scan1[x_edge]['edgevi'][0]],
                          [sub_scan2[x_edge]['edgevi'][1],sub_scan2[x_edge]['edgevi'][0]])

            # scan2 to scan1
            _combine_path(sub_scan2[x_edge]['s_e']['s'],
                          sub_scan1[x_edge]['s_e']['m'],
                          sub_scan2['edge'][i_edge],
                          sub_scan1['edge'][i_edge],
                          sub_scan2[x_edge]['edgepos2d'],
                          [sub_scan1[x_edge]['edgepos2d'][1], sub_scan1[x_edge]['edgepos2d'][0]],
                          scan2, scan1,
                          sub_scan2[x_edge]['edgevi'],
                          sub_scan1[x_edge]['edgevi'])

            _combine_path(sub_scan2[x_edge]['e_s']['s'],
                          sub_scan1[x_edge]['e_s']['m'],
                          (sub_scan2['edge'][i_edge][1],sub_scan2['edge'][i_edge][0]),
                          (sub_scan1['edge'][i_edge][1],sub_scan1['edge'][i_edge][0]),
                          [sub_scan2[x_edge]['edgepos2d'][1], sub_scan2[x_edge]['edgepos2d'][0]],
                          sub_scan1[x_edge]['edgepos2d'],
                          scan2, scan1,
                          [sub_scan2[x_edge]['edgevi'][1],sub_scan2[x_edge]['edgevi'][0]],
                          [sub_scan1[x_edge]['edgevi'][1],sub_scan1[x_edge]['edgevi'][0]])

    inputs = json.load(open(args.input_file_path))

    outputs = []

    num_graph = set()
    len_ins = 0
    for i in inputs:
        num_graph.add(i['scan'])
        len_ins += len(i['instructions'])
        outputs.append({
            'distance':     i['distance'],
            'scan':         i['scan'],
            'path_id':      i['path_id'],
            'path':         i['path'],
            'heading':      i['heading'],
            'instructions': i['instructions']
        })

    num_new_graph = set()
    len_new_ins = 0
    random.shuffle(new_paths)
    for i in new_paths:
        if len_new_ins >= len_ins*0.25:
            break
        num_new_graph.add(i['scan'])
        len_new_ins += len(i['instructions'])
        outputs.append({
            'distance':     i['distance'],
            'scan':         i['scan'],
            'path_id':      i['path_id'],
            'path':         i['path'],
            'heading':      i['heading'],
            'instructions': i['instructions'],
            'second':       i['second']
        })

    len_tol_ins = 0
    for i in outputs:
        len_tol_ins += len(i['instructions'])

    with open(args.output_file_path, 'w') as f:
        json.dump(outputs, f, indent=2, sort_keys=True, separators=(',', ': '))
    print('******Final Results********')
    print('  Number of graph:         {}'.format(len(num_graph)))
    print('  Path:                    {}'.format(len(inputs)))
    print('  Instructions:            {}'.format(len_ins))
    print('  Number of new graph:     {}'.format(len(num_new_graph)))
    print('  New path:                {}'.format(len(new_paths)))
    print('  New instructions:        {}'.format(len_new_ins))
    print('  Total path:              {}'.format(len(outputs)))
    print('  Total instructions:      {}'.format(len_tol_ins))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--connections_dir',
      dest='connections_dir',
      required=True,
      help='Path to the Matterport simulator connection data.')
  parser.add_argument(
      '--input_file_path',
      dest='input_file_path',
      required=True,
      help='Path to read the R2R input data.')
  parser.add_argument(
      '--mid_file_path',
      dest='mid_file_path',
      required=True,
      help='SubPath to write the SUBR2R output data.')
  parser.add_argument(
      '--output_file_path',
      dest='output_file_path',
      required=True,
      help='Path to write the RER2R output data.')

  ARG = parser.parse_args()
  #main(ARG)
  generate_RER2R(ARG)

