from sys import argv
import re
from matplotlib import pyplot as pl, colors
import numpy as np
import pandas as pd

R_AVG_RANGE = 100
FILENAMES = argv[1:] if len(argv) > 1 else [input("Filename: ")]
# FILENAMES = argv[1:] if len(argv) > 1 else ["mnist_90_1.txt"]
HUES = np.linspace(0, 1, len(FILENAMES), endpoint=False)

for i, filename in enumerate(FILENAMES):
	filename_short = filename[-filename[::-1].find("\\"):-4]
	with open(filename,"r") as f:
		logfile = f.read()

	costs_str = re.findall("cost [0-9]\.[0-9]*?,", logfile)
	costs = []
	for cost in costs_str:
		costs.append(float(cost[5:-1]))

	data = {
		# could use log(costs)
		"cost": costs,
	}
	dataframe = pd.DataFrame(data=data)
	data_avg = dataframe.rolling(R_AVG_RANGE).mean()

	pl.plot(np.arange(len(costs)), data_avg["cost"], color=colors.hsv_to_rgb((HUES[i], 1, 0.8)), label=filename_short)
	# ...why can't i specify resolution in pixels?? better results when i MANUALLY save the graph at 1080p...
	#pl.savefig(f"{filename_short}_perc_{R_AVG_RANGE}.png", dpi=1200, bbox_inches="tight")

pl.title("Cost reduction over batches")
pl.xlabel("Training batches")
pl.ylabel("Cost")
pl.legend(loc="best")
pl.grid(True)
pl.show()