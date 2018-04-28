from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.draw import line_aa
from MSTME import *
from UnionFind import *

import numpy as np
import 	math

if __name__ == "__main__":

	t = rgb2gray(imread("example.png"))
	height = t.shape[0]
	width = t.shape[1]
	entropy_lambda = -10 
	vertices = list()
	 


	print("Image Shape is : " + str(t.shape))

	for h in range(height):
		for w in range(width):
			if t[h, w] == 0:
				vertices.append((h, w))


	vsize = len(vertices)
	print(vsize)	 
	vdistances = np.zeros([vsize, vsize])

	for index1 in range(vsize): 
		for index2 in range(vsize - index1 - 1):
			ind1 = index1
			ind2 = index1 + index2 + 1
			v1 = vertices[ind1]	 
			v2 = vertices[ind2]
			d = ((v2[1] - v1[1]) ** 2 + (v2[0] - v1[0]) ** 2) ** (1 / 2)
			vdistances[ind1, ind2] = d
			vdistances[ind2, ind1] = d

	mstme = MSTME(vdistances)
	mst_edges = mstme.build()

	for mst_edge in mst_edges:
		v1 = vertices[mst_edge[0]]
		v2 = vertices[mst_edge[1]]
		rr, cc, val = line_aa(v1[0], v1[1], v2[0], v2[1])
		t[rr, cc] = val

	imsave(str(entropy_lambda) + "example_output.png", t)
