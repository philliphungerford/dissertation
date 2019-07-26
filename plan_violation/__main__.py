#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Deep learning models for predicting plan violations in radiotherapy data
"""
# =============================================================================
# Run models
# =============================================================================

if __name__ == '__main__':
	pointnet_full()
	pointnet_basic()
	pointnet_basic_l()
	cnn_3d()