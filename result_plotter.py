import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, getopt

matplotlib.use("Agg")


def main_plotter():
    if len(sys.argv) < 4:
        print(
            "Usage: {0} chart_title output.png input1.csv [input2.csv input3.csv ...]".format(
                sys.argv[0]
            )
        )
        sys.exit(2)


chart_title = sys.argv[1]
output_file = sys.argv[2]
input_files = sys.argv[3:]

x_label = "Size (m0)"
y_label = "Performance (Gflop/s)"

ymax = 1

figure, ax = plt.subplots()

for input_file in input_files:
    df = pd.read_csv(input_file)
    xsize = df["m0"]
    res = df["gflops"]
    ax.plot(xsize, res, label=input_file)
    ymax = max(ymax, max(res))

ax.set(xlabel=x_label, ylabel=y_label, title=chart_title, ylim=[0, ymax * 1.5])

plt.legend()
plt.grid()
plt.savefig(output_file)


if __name__ == "__main__":
    main_plotter()
