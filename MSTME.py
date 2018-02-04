from UnionFind import *
import numpy as np
import 	math

class MSTME:

	def __init__(self, distances, entropy_lambda=1):
		self.distances = distances
		self.entropy_lambda=entropy_lambda

	def LOSS(self, edges, distances, entropy_lambda = 1):
		edge_distances = [distances[edge] for edge in edges]
		return np.sum(edge_distances) - entropy_lambda * self.__H(edge_distances)

	def __H(self, edge_distances):
		frequencies = dict()
		for edge in edge_distances:
			if edge not in frequencies:
				frequencies[edge] = 1
			else:
				frequencies[edge] = frequencies[edge] + 1

		edges_size = len(edge_distances)
		return -1 * np.sum([frequency/edges_size * math.log(frequency/edges_size, 2) for frequency in frequencies])
	def build(self):

		mst_edges = list()
		vsize = self.distances.shape[0]
		uf = UnionFind(vsize)

		for i in range(vsize - 1):
			best_loss = 2 ** 32
			best_edge = None 
			for index1 in range(vsize): 
				for index2 in range(vsize - index1 - 1):
					ind1 = index1
					ind2 = index1 + index2 + 1

					if not uf.find(ind1, ind2):
						temp_loss = self.LOSS(mst_edges + [(ind1, ind2)], self.distances, entropy_lambda=self.entropy_lambda)

						if temp_loss < best_loss:
							best_loss=temp_loss
							best_edge = (ind1, ind2)

			mst_edges = mst_edges + [best_edge]
			print(mst_edges)
			uf.union(best_edge[0], best_edge[1])
		return mst_edges


