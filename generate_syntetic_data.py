import numpy as np
import os
import pandas as pd
import random

FILE_NAME = 'square_data.xlsx'


def get_eccentricity_ranges(bins):
    eccentricity_bins = np.linspace(0, 1.0, bins + 1)  # ox
    ranges = []

    for i in range(len(eccentricity_bins)):
        if (i != len(eccentricity_bins) - 1):
            ranges.append(f"{round(eccentricity_bins[i], 2)}-{round(eccentricity_bins[i + 1], 2)}")

    return ranges


def get_square_ranges(bins):
    eccentricity_bins = np.linspace(5, 8000, bins + 1)  # ox
    ranges = []

    for i in range(len(eccentricity_bins)):
        if (i != len(eccentricity_bins) - 1):
            ranges.append(f"{round(eccentricity_bins[i], 2)}nm^2-{round(eccentricity_bins[i + 1], 2)}nm^2")

    return ranges


def generate_particles(bins, nums):  # oy
    n = 0
    values = []

    while n < nums:
        n += 1
        current_values = []

        for i in range(bins):
            num_particles = random.randint(0, 50)
            current_values.append(num_particles)
        values.append(current_values)

    return values


def write_to_excel_file(values, ranges, shift):
    df = pd.DataFrame(values, columns=ranges)

    with pd.ExcelWriter(FILE_NAME, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, startcol=shift, index=False)


def create_empty_file():
    if os.path.exists(FILE_NAME):
        os.remove(FILE_NAME)

    pd.DataFrame().to_excel(FILE_NAME, index=False)


create_empty_file()

bins_ = 10  # num of bins
nums_ = 200  # num of data
current_shift = 0

for i in range(3):  # three different ranges
    get_ranges = get_square_ranges(bins_)  # a convenient form for excel
    get_values = generate_particles(bins_, nums_)

    write_to_excel_file(get_values, get_ranges, current_shift)

    current_shift += len(get_ranges) + 2
    bins_ += 5





