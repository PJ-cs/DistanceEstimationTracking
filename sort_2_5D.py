
from __future__ import print_function

import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def sim_score(bb_test, bb_gt, alpha_iou=0.5, beta_distz=0.5, max_dist=1):
  """
  Computes Similarity score between two bboxes in the form [x1,y1,x2,y2, z]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  dist_z = ((max_dist - np.abs(bb_test[...,4]-bb_gt[..., 4]))/max_dist).clip(0.,1.)
  o = (alpha_iou * (wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
       + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)) 
       + beta_distz * dist_z )/ (alpha_iou + beta_distz)                                         
  return(o)  



def convert_bbox_depth(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2, z] and returns in the form
    [x,y,s,r,z] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio and z is the depth
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r, bbox[4]]).reshape((5, 1))


def convert_x_to_bbox_depth(x):
  """
  Takes a bounding box in the centre form [x,y,s,r, z] and returns it in the form
    [x1,y1,x2,y2, z] where x1,y1 is the top left and x2,y2 is the bottom right and z is depth
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  
  return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,x[4]]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # new StateVektor: x  [u,v,s,r,z,u',v',s',z']
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=9, dim_z=5) 
    self.kf.F = np.array([[1,0,0,0,0,1,0,0,0],
                          [0,1,0,0,0,0,1,0,0],
                          [0,0,1,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,1],
                          [0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,1]])
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0]])

    
    self.kf.R[2:,2:] *= 10. # measurment noise
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities, covariance matrix
    self.kf.P *= 10.
    self.kf.Q[-2,-2] *= 0.01 #process noise
    self.kf.Q[5:,5:] *= 0.01
    #print(self.kf.R, self.kf.P, self.kf.Q, sep="\n") # TODO remove
    
    self.kf.x[:5] = convert_bbox_depth(bbox)
    
    
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_depth(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if future scale (area) is <= 0, set scale velocity to 0
    if((self.kf.x[7]+self.kf.x[2])<=0):
      self.kf.x[7] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox_depth(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox_depth(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3, alpha_iou=0.5, beta_distz=0.5, max_dist=1):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int) 

  iou_matrix = sim_score(detections, trackers, alpha_iou, beta_distz, max_dist)
  

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort2_5D(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, alpha_iou=0.5, max_dist=1):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.alpha_iou = alpha_iou
    self.beta_distz = 1.-alpha_iou
    self.max_dist = max_dist
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,depth],[x1,y1,x2,y2,depth],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = pos[:5]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold, self.alpha_iou, self.beta_distz, self.max_dist) 

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,6))
