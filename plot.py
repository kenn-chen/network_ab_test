#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def result_plot():
	graphs = ['growing_network', 'wiki-Vote', 'soc-Epinions1', 'soc-Slashdot0811']
	#graphs = ["wiki-Vote"]
	for graph in graphs:
		plt.figure(graph)
		plt.subplots_adjust(hspace = 0.4)
		plt.subplots_adjust(wspace = 0.3)
		data = defaultdict(lambda: defaultdict(list))
		with open("results/"+graph) as fin:
			for line in fin:
				lambda1, lambda2, rmse, _, _, method = line.split()
				if method == "new":
					method = "Proposed"
				elif method == "baseline1":
					method = "Baseline1"
				elif method == "baseline2":
					continue
				elif method == "baseline3":
					method = "Baseline2"
				data[float(lambda2)][method].append((float(lambda1), rmse))

		cnt = 0
		for lambda2 in sorted(data.keys()):
			cnt += 1
			ax = plt.subplot("22%d" % cnt)
			ax.set_title(r"$\lambda_2$=%.2f" % lambda2)
			ax.set_xlabel(r'$\lambda_1$')
			ax.set_ylabel('RMSE')
			for method in sorted(data[lambda2].keys()):
				points = data[lambda2][method]
				points.sort(key=lambda x: x[0])
				plt.plot(*zip(*points), marker='o', linestyle="--", label=method)
			legend = plt.legend(loc='upper left', prop={'size':10})
			plt.savefig(graph+'.svg', format='svg', dpi=100)

	#plt.show()


def community_plot():
	soc_Epinions1 = [19540, 2990, 2307, 1890, 1839, 1632, 1364, 1206, 1122, 1036, 1019, 942, 916, 856, 762, 755, 744, 713, 697, 696, 669, 652, 622, 609, 599, 595, 577, 575, 567, 550, 549, 544, 534, 516, 513, 511, 510, 508, 502, 499, 499, 493, 476, 453, 451, 447, 444, 440, 429, 428, 424, 423, 418, 414, 412, 412, 403, 403, 386, 376, 371, 369, 368, 364, 364, 363, 363, 359, 357, 355, 354, 352, 351, 349, 341, 340, 337, 336, 331, 329, 326, 324, 324, 316, 312, 311, 305, 303, 298, 296, 290, 287, 282, 270, 269, 268, 255, 252, 245, 222, 213]
	soc_Slashdot0811 = [7874, 4144, 2917, 2450, 1655, 1553, 1464, 1375, 1118, 895, 891, 881, 877, 869, 853, 850, 849, 824, 822, 820, 798, 787, 778, 767, 754, 735, 734, 722, 702, 695, 690, 690, 683, 680, 677, 672, 663, 638, 629, 621, 618, 615, 602, 590, 590, 574, 567, 565, 564, 557, 556, 551, 549, 548, 544, 543, 537, 527, 521, 513, 513, 513, 511, 510, 509, 508, 505, 477, 477, 476, 470, 467, 464, 462, 459, 459, 457, 456, 453, 445, 444, 440, 438, 431, 430, 425, 420, 419, 417, 407, 400, 398, 396, 396, 389, 380, 380, 379, 365, 350, 338]
	wiki_Vote = [388, 211, 175, 165, 160, 148, 131, 124, 116, 115, 113, 111, 108, 105, 103, 101, 96, 96, 96, 94, 92, 90, 88, 84, 82, 81, 78, 77, 76, 75, 75, 75, 74, 70, 69, 68, 68, 67, 67, 66, 64, 64, 63, 61, 59, 59, 58, 58, 58, 58, 57, 57, 56, 56, 55, 54, 53, 52, 52, 51, 51, 50, 50, 50, 49, 48, 47, 47, 46, 46, 46, 45, 44, 44, 43, 43, 43, 43, 43, 43, 43, 42, 42, 41, 40, 40, 40, 39, 39, 39, 38, 38, 37, 35, 34, 34, 33, 32, 32, 27, 26]
	growing_network = [1732, 654, 572, 568, 524, 521, 521, 449, 424, 419, 380, 335, 313, 308, 293, 286, 262, 261, 237, 225, 223, 221, 219, 217, 215, 213, 204, 203, 200, 197, 194, 188, 188, 183, 181, 177, 176, 173, 172, 170, 164, 162, 159, 158, 157, 150, 150, 146, 142, 140, 139, 138, 136, 136, 135, 134, 133, 131, 131, 128, 128, 127, 126, 125, 125, 125, 121, 120, 119, 117, 116, 115, 114, 113, 112, 106, 105, 105, 104, 100, 100, 99, 98, 97, 94, 93, 91, 90, 90, 87, 86, 80, 77, 75, 72, 67, 64, 56, 47, 47]
	data = [("soc-Epinions1", soc_Epinions1), ("soc-Slashdot0811", soc_Slashdot0811), ("wiki-Vote", wiki_Vote), ("growing_network", growing_network)]
	plt.xlabel('Cluster ID')
	plt.ylabel('Cluster Size')
	for name, clusters in data:
		plt.plot(clusters[:20], label=name, marker='s', linewidth=1.5, linestyle=":")
	legend = plt.legend(loc='upper right')
	#plt.show()
	plt.savefig('community.svg', format='svg', dpi=100)

if __name__ == "__main__":
	result_plot()
	#community_plot()
