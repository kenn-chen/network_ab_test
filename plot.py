from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

graphs = ['growing_network', 'wiki-Vote', 'soc-Epinions1', 'soc-Slashdot0811']
#graphs = ["wiki-Vote"]
for graph in graphs:
	plt.figure(graph)
	plt.subplots_adjust(hspace = 0.4)
	data = defaultdict(lambda: defaultdict(list))
	with open("results/"+graph) as fin:
		for line in fin:
			lambda1, lambda2, rmse, _, _, method = line.split()
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
		plt.legend(loc='upper left')

plt.show()
