# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for loading and drawing graphs of the houses."""

from __future__ import print_function

import json
import os
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm

def save_single(root,connections_file1,connections_file2,scan1,scan2,sub_scan1,sub_scan2,ith_edge='first_edge'):

  with open(connections_file1) as f:
    lines1 = json.load(f)

  with open(connections_file2) as f:
    lines2 = json.load(f)

  lines1_num = len(lines1)
  lines2_num = len(lines2)
  i2n = dict()
  n2i = dict()
  lines = list()
  for i,v in enumerate(lines1):
      v['visible'] = v['visible'] + [False]*lines2_num
      v['unobstructed'] = v['unobstructed'] + [False]*lines2_num
      i2n[i] = v['image_id']
      n2i[v['image_id']] = i
      lines.append(v)

  pos3d = sub_scan1[ith_edge]['edgepos3d'][1] - sub_scan2[ith_edge]['edgepos3d'][1]

  for i,v in enumerate(lines2):
      v['visible'] = [False]*lines1_num + v['visible']
      v['unobstructed'] = [False]*lines1_num + v['unobstructed']
      v['pose'][3] = v['pose'][3] + pos3d[0]
      v['pose'][7] = v['pose'][7] + pos3d[1]
      v['pose'][11] = v['pose'][11] + pos3d[2]
      i2n[i+lines1_num] = v['image_id']
      n2i[v['image_id']] = i+lines1_num
      lines.append(v)

  out_file = os.path.join(root, '{}+{}_connectivity.json'.format(scan1,scan2))
  edge_file = os.path.join(root, '{}+{}_edge.json'.format(scan1, scan2))
  se1_1,se1_2 = sub_scan1['edge']
  se2_1,se2_2 = sub_scan2['edge']
  s1_1,e1_1 = se1_1
  s1_2,e1_2 = se1_2
  s2_1,e2_1 = se2_1
  s2_2,e2_2 = se2_2

  def _fixnode(s1,e1,e2):
      lines[n2i[s1]]['visible'][n2i[e1]] = False
      lines[n2i[s1]]['unobstructed'][n2i[e1]] = False
      lines[n2i[s1]]['visible'][n2i[e2]] = True
      lines[n2i[s1]]['unobstructed'][n2i[e2]] = True
  def _edgeheading(s,e):
      s_x = lines[n2i[s]]['pose'][3]
      s_y = lines[n2i[s]]['pose'][7]
      e_x = lines[n2i[e]]['pose'][3]
      e_y = lines[n2i[e]]['pose'][7]
      x,y = e_x-s_x, e_y-s_y
      return round(np.arctan2(x, y) % (2 * np.pi), 3)
      # return round(np.arctan2(y, x) % (2 * np.pi), 3)
  def _edgeout(s1,e1,s2,e2):
      edge_out = {'edge':
                      [[s1,e1],
                       [s2,e2]],
                  'heading':
                      [[_edgeheading(s1,e2),_edgeheading(e1,s2)],
                       [_edgeheading(s2,e1),_edgeheading(e2,s1)]]}
      return edge_out


  if ith_edge=='first_edge':
      _fixnode(s1_1, e1_1, e2_1)
      _fixnode(e1_1, s1_1, s2_1)
      _fixnode(s2_1, e2_1, e1_1)
      _fixnode(e2_1, s2_1, s1_1)
      edge_out = _edgeout(s1_1,e1_1,s2_1,e2_1)
  else:
      _fixnode(s1_2, e1_2, e2_2)
      _fixnode(e1_2, s1_2, s2_2)
      _fixnode(s2_2, e2_2, e1_2)
      _fixnode(e2_2, s2_2, s1_2)
      edge_out = _edgeout(s1_2, e1_2,s2_2, e2_2)
  if not os.path.isdir(root):
      os.makedirs(root)
  with open(out_file,'w') as f:
    json.dump(lines,f)
  with open(edge_file,'w') as f:
    json.dump(edge_out,f)

  return

def save(root,connections_file1,connections_file2,scan1,scan2,sub_scan1,sub_scan2):

  with open(connections_file1) as f:
    lines1 = json.load(f)

  with open(connections_file2) as f:
    lines2 = json.load(f)

  lines1_num = len(lines1)
  lines2_num = len(lines2)
  i2n = dict()
  n2i = dict()
  lines = list()
  for i,v in enumerate(lines1):
      v['visible'] = v['visible'] + [False]*lines2_num
      v['unobstructed'] = v['unobstructed'] + [False]*lines2_num
      i2n[i] = v['image_id']
      n2i[v['image_id']] = i
      lines.append(v)

  for i,v in enumerate(lines2):
      v['visible'] = [False]*lines1_num + v['visible']
      v['unobstructed'] = [False]*lines1_num + v['unobstructed']
      i2n[i+lines1_num] = v['image_id']
      n2i[v['image_id']] = i+lines1_num
      lines.append(v)

  out_file = os.path.join(root, '{}+{}_connectivity.json'.format(scan1,scan2))
  se1_1,se1_2 = sub_scan1['edge']
  se2_1,se2_2 = sub_scan2['edge']
  s1_1,e1_1 = se1_1
  s1_2,e1_2 = se1_2
  s2_1,e2_1 = se2_1
  s2_2,e2_2 = se2_2

  def _fixnode(s1,e1,e2):
      lines[n2i[s1]]['visible'][n2i[e1]] = False
      lines[n2i[s1]]['unobstructed'][n2i[e1]] = False
      lines[n2i[s1]]['visible'][n2i[e2]] = True
      lines[n2i[s1]]['unobstructed'][n2i[e2]] = True

  _fixnode(s1_1, e1_1, e2_1)
  _fixnode(e1_1, s1_1, s2_1)
  _fixnode(s1_2, e1_2, e2_2)
  _fixnode(e1_2, s1_2, s2_2)

  _fixnode(s2_1, e2_1, e1_1)
  _fixnode(e2_1, s2_1, s1_1)
  _fixnode(s2_2, e2_2, e1_2)
  _fixnode(e2_2, s2_2, s1_2)

  with open(out_file,'w') as f:
    json.dump(lines,f)

  return

def load(connections_file):
  """Loads a networkx graph for a given scan.

  Args:
    connections_file: A string with the path to the .json file with the
      connectivity information.
  Returns:
    A networkx graph.
  """
  with open(connections_file) as f:
    lines = json.load(f)
    nodes = np.array([x['image_id'] for x in lines])
    matrix = np.array([x['unobstructed'] for x in lines])
    mask = [x['included'] for x in lines]
    matrix = matrix[mask][:, mask]
    nodes = nodes[mask]
    pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
    pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}

  graph = nx.from_numpy_matrix(matrix)
  graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
  nx.set_node_attributes(graph, pos2d, 'pos2d')
  nx.set_node_attributes(graph, pos3d, 'pos3d')

  weight2d = {(u, v): norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
  weight3d = {(u, v): norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
  nx.set_edge_attributes(graph, weight2d, 'weight2d')
  nx.set_edge_attributes(graph, weight3d, 'weight3d')

  return graph


def draw(graph, predicted_path, reference_path, output_filename, **kwargs):
  """Generates a plot showing the graph, predicted and reference paths.

  Args:
    graph: A networkx graph.
    predicted_path: A list with the ids of the nodes in the predicted path.
    reference_path: A list with the ids of the nodes in the reference path.
    output_filename: A string with the path where to store the generated image.
    **kwargs: Key-word arguments for aesthetic control.
  """
  plt.figure(figsize=(10, 10))
  ax = plt.gca()
  pos = nx.get_node_attributes(graph, 'pos2d')

  # Zoom in.
  # xs = [pos[node][0] for node in predicted_path + reference_path]
  # ys = [pos[node][1] for node in predicted_path + reference_path]
  # min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
  # center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
  # zoom_margin = kwargs.get('zoom_margin', 1.3)
  # max_range = zoom_margin * max(max_x - min_x, max_y - min_y)
  # half_range = max_range / 2
  # ax.set_xlim(center_x - half_range, center_x + half_range)
  # ax.set_ylim(center_y - half_range, center_y + half_range)

  # Background graph.
  nx.draw(graph,
          pos,
          edge_color=kwargs.get('background_edge_color', 'lightgrey'),
          node_color=kwargs.get('background_node_color', 'lightgrey'),
          node_size=kwargs.get('background_node_size', 60),
          width=kwargs.get('background_edge_width', 0.5))

  # Prediction graph.
  predicted_path_graph = nx.DiGraph()
  predicted_path_graph.add_nodes_from(predicted_path)
  # predicted_path_graph.add_edges_from(
  #     zip(predicted_path[:-1], predicted_path[1:]))
  nx.draw(predicted_path_graph,
          pos,
          arrowsize=kwargs.get('prediction_arrowsize', 15),
          edge_color=kwargs.get('prediction_edge_color', 'red'),
          node_color=kwargs.get('prediction_node_color', 'red'),
          node_size=kwargs.get('prediction_node_size', 130),
          width=kwargs.get('prediction_edge_width', 2.0))

  # Reference graph.
  reference_path_graph = nx.DiGraph()
  reference_path_graph.add_nodes_from(reference_path)
  # reference_path_graph.add_edges_from(
  #     zip(reference_path[:-1], reference_path[1:]))
  nx.draw(reference_path_graph,
          pos,
          arrowsize=kwargs.get('reference_arrowsize', 15),
          edge_color=kwargs.get('reference_edge_color', 'dodgerblue'),
          node_color=kwargs.get('reference_node_color', 'dodgerblue'),
          node_size=kwargs.get('reference_node_size', 130),
          width=kwargs.get('reference_edge_width', 2.0))

  # Intersection graph.
  intersection_path_graph = nx.DiGraph()
  common_nodes = set(predicted_path_graph.nodes.keys()).intersection(
      set(reference_path_graph.nodes.keys()))
  intersection_path_graph.add_nodes_from(common_nodes)
  common_edges = set(predicted_path_graph.edges.keys()).intersection(
      set(reference_path_graph.edges.keys()))
  intersection_path_graph.add_edges_from(common_edges)
  nx.draw(intersection_path_graph,
          pos,
          arrowsize=kwargs.get('intersection_arrowsize', 15),
          edge_color=kwargs.get('intersection_edge_color', 'limegreen'),
          node_color=kwargs.get('intersection_node_color', 'limegreen'),
          node_size=kwargs.get('intersection_node_size', 130),
          width=kwargs.get('intersection_edge_width', 2.0))

  plt.savefig(output_filename)
  plt.show()
  plt.close()

def edge2point(edge):
    if(type(edge[0])!=tuple):
        return edge,None
    point=set()
    for s,e in edge:
        point.add(s)
        point.add(e)
    return point,edge


def draw_edge(graph, predicted_path, reference_path, output_filename, **kwargs):
  """Generates a plot showing the graph, predicted and reference paths.

  Args:
    graph: A networkx graph.
    predicted_path: A list with the ids of the nodes in the predicted path.
    reference_path: A list with the ids of the nodes in the reference path.
    output_filename: A string with the path where to store the generated image.
    **kwargs: Key-word arguments for aesthetic control.
  """

  plt.figure(figsize=(10, 10))
  ax = plt.gca()
  pos = nx.get_node_attributes(graph, 'pos2d')

  # Zoom in.
  # xs = [pos[node][0] for node in predicted_path + reference_path]
  # ys = [pos[node][1] for node in predicted_path + reference_path]
  # min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
  # center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
  # zoom_margin = kwargs.get('zoom_margin', 1.3)
  # max_range = zoom_margin * max(max_x - min_x, max_y - min_y)
  # half_range = max_range / 2
  # ax.set_xlim(center_x - half_range, center_x + half_range)
  # ax.set_ylim(center_y - half_range, center_y + half_range)

  # Background graph.
  nx.draw(graph,
          pos,
          edge_color=kwargs.get('background_edge_color', 'lightgrey'),
          node_color=kwargs.get('background_node_color', 'lightgrey'),
          node_size=kwargs.get('background_node_size', 60),
          width=kwargs.get('background_edge_width', 0.5))

  # Prediction graph.
  predicted_path, predicted_path_edge = edge2point(predicted_path)
  predicted_path_graph = nx.Graph()
  predicted_path_graph.add_nodes_from(predicted_path)
  if predicted_path_edge != None:
      predicted_path_graph.add_edges_from(predicted_path_edge)
  nx.draw(predicted_path_graph,
          pos,
          arrowsize=kwargs.get('prediction_arrowsize', 15),
          edge_color=kwargs.get('prediction_edge_color', 'red'),
          node_color=kwargs.get('prediction_node_color', 'red'),
          node_size=kwargs.get('prediction_node_size', 130),
          width=kwargs.get('prediction_edge_width', 2.0))

  # Reference graph.
  reference_path, reference_path_edge = edge2point(reference_path)
  reference_path_graph = nx.Graph()
  reference_path_graph.add_nodes_from(reference_path)
  if reference_path_edge != None:
      reference_path_graph.add_edges_from(reference_path_edge)
  nx.draw(reference_path_graph,
          pos,
          arrowsize=kwargs.get('reference_arrowsize', 15),
          edge_color=kwargs.get('reference_edge_color', 'dodgerblue'),
          node_color=kwargs.get('reference_node_color', 'dodgerblue'),
          node_size=kwargs.get('reference_node_size', 130),
          width=kwargs.get('reference_edge_width', 2.0))

  # Intersection graph.
  intersection_path_graph = nx.Graph()
  common_nodes = set(predicted_path_graph.nodes.keys()).intersection(
      set(reference_path_graph.nodes.keys()))
  intersection_path_graph.add_nodes_from(common_nodes)
  common_edges = set(graph.subgraph(common_nodes).edges.keys()).intersection(
      set(reference_path_graph.edges.keys()))
  intersection_path_graph.add_edges_from(common_edges)
  nx.draw(intersection_path_graph,
          pos,
          arrowsize=kwargs.get('intersection_arrowsize', 15),
          edge_color=kwargs.get('intersection_edge_color', 'limegreen'),
          node_color=kwargs.get('intersection_node_color', 'limegreen'),
          node_size=kwargs.get('intersection_node_size', 130),
          width=kwargs.get('intersection_edge_width', 2.0))

  plt.savefig(output_filename)
  plt.show()
  plt.close()
  return common_edges
