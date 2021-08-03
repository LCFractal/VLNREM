''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args

from utils import load_datasets, load_nav_graphs, Tokenizer

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        self.seconds = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)
            self.seconds.append(None)

    def _make_id(self, scanId, viewpointId):
        if '+' in scanId:
            sc=scanId.split('+')
            return [sc[0]+'_'+viewpointId, sc[1]+'_'+viewpointId]
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings, seconds, graphs):
        self.graphs = graphs
        for i, (scanId, viewpointId, heading, second) in enumerate(zip(scanIds, viewpointIds, headings, seconds)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.seconds[i] = second
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            second = self.seconds[i]
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                if type(long_id) == list:
                    if long_id[0] in list(self.features.keys()):
                        feature = self.features[long_id[0]]
                        scanId = state.scanId.split('+')[0]
                        second_scanId = state.scanId.split('+')[1]
                    else:
                        feature = self.features[long_id[1]]
                        scanId = state.scanId.split('+')[1]
                        second_scanId = state.scanId.split('+')[0]
                    if state.location.viewpointId in second['edge1']+second['edge2']:
                        vii = (second['edge1']+second['edge2']).index(state.location.viewpointId)
                        if vii<2:
                            vi = second['vi1'][vii%2]
                            ex_vi = second['vi2'][vii%2]
                            ex_feature = self.features[self._make_id(second['scan2'], second['edge2'][vii % 2])]
                            vi = second['vri1']
                        else:
                            vi = second['vi2'][vii%2]
                            ex_vi = second['vi1'][vii % 2]
                            ex_feature = self.features[self._make_id(second['scan1'], second['edge1'][vii % 2])]
                            vi = second['vri2']
                        feature = np.array(feature)
                        ex_feature = np.array(ex_feature)
                        # vv_list = [vi]
                        vv_list = [(vi - 1) % 12, vi, (vi+1) % 12]
                        # vv_list = [(vi - 2) % 12,(vi - 1) % 12, vi, (vi+1)%12, (vi+2)%12]
                        for vv_i,vv in enumerate(vv_list):
                            # vv_mask = abs(vv_i-len(vv_list)/2)/(len(vv_list)/2+1)
                            # vv_mask = abs(vv_i-int(len(vv_list)/2))/(int(len(vv_list)/2+1))
                            vv_mask = 0
                            ex_vv = (vv+ex_vi-vi)%12
                            feature[vv] = ex_feature[ex_vv]*(1-vv_mask)+feature[vv]*vv_mask
                            feature[vv + 12] = ex_feature[ex_vv + 12]*(1-vv_mask)+feature[vv + 12]*vv_mask
                            feature[vv + 24] = ex_feature[ex_vv + 24]*(1-vv_mask)+feature[vv + 24]*vv_mask
                    else:
                        # false negative
                        if args.falseNav:
                            # print('falseNav')
                            feature = np.array(feature)
                            second_view_list = list(self.graphs[second_scanId])
                            choice_view = random.choice(second_view_list)
                            choice_long_id = self._make_id(second_scanId, choice_view)
                            choice_feature = self.features[choice_long_id]
                            choice_feature_num = 2
                            have_choice_num = 0
                            choice_view_num_list = list(range(0,12))
                            random.shuffle(choice_view_num_list)
                            for vi in choice_view_num_list:
                                if vi in [(state.viewIndex-1)%12,state.viewIndex%12,(state.viewIndex+1)%12]:
                                    continue
                                have_choice_num+=1

                                feature[vi] = choice_feature[vi]
                                feature[vi + 12] = choice_feature[vi + 12]
                                feature[vi + 24] = choice_feature[vi + 24]
                                if have_choice_num >= choice_feature_num:
                                    break
                else:
                    if second==None:
                        feature = self.features[long_id]
                    else:
                        feature = self.features[long_id]
            else:
                feature = None
            feature_states.append((feature, state, self.seconds[i]))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        if item['scan'].split('+')[0] not in self.env.featurized_scans:
                            continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state, second) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        seconds = [item['second'] if 'second' in list(item.keys()) else None for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings, seconds,self.graphs)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


