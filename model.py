from __future__ import division
import matplotlib
from math import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse


V_R = 10.0
l = 1.50
Dt = 0.05

def diff_eqn(state, delta, V_R = V_R):

	x_g, y_g, theta = state
	xdot = V_R*cos(theta)
	ydot = V_R*sin(theta)
	thetadot = 1*delta #V_R/l

	return np.array([xdot, ydot, thetadot])

def model(state, delta, V_R = V_R):
	'''
	x_t+1 =  f(x_t, u_t)
	Input:
	state_t: state of the system at time t
	delta_f: the angular of the front wheels at time t
	dt:
	Output:
	state_t = the next state
	'''

	x_g, y_g, theta = state
	k1 = diff_eqn(state, delta, V_R = V_R)
	k2 = diff_eqn(state + (k1/2)*Dt, delta, V_R = V_R)
	k3 = diff_eqn(state + (k2/2)*Dt, delta, V_R = V_R)
	k4 = diff_eqn(state + (k3)*Dt, delta, V_R = V_R)
	Xnew = state + (k1 + 2*k2 + 2*k3 + k4)*Dt/6

	state = [float(Xnew[0]), float(Xnew[1]), float(Xnew[2])%(2*np.pi)]

	return state


def test_module():
	state = np.array([0, 0, 0])
	delta = 0.1
	state = model(state, delta)
	print (state)

#test_module()
