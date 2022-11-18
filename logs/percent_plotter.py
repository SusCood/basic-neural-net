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

	batch_size = int(re.search("/[0-9]* correct", logfile)[0][1:4])
	perc_str = re.findall(", [0-9]{1,3}/" + str(batch_size), logfile)
	percents = []
	for percent in perc_str:
		percents.append(round(int(percent[2:-4])/batch_size * 100, 2))

	data = {
		"percent": percents,
	}
	dataframe = pd.DataFrame(data=data)
	data_avg = dataframe.rolling(R_AVG_RANGE).mean()

	pl.plot(np.arange(len(percents)), data_avg["percent"], color=colors.hsv_to_rgb((HUES[i], 1, 0.8)), label=filename_short)
	# ...why can't i specify resolution in pixels?? better results when i MANUALLY save the graph at 1080p...
	#pl.savefig(f"{filename_short}_perc_{batch_size}_{R_AVG_RANGE}.png", dpi=1200, bbox_inches="tight")

pl.title("Accuracy over batches")
pl.xlabel("Training batches")
pl.ylabel("Accuracy (%)")
pl.legend(loc="best")
pl.grid(True)
pl.show()