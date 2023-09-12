import pandas as pd
import csv
from datetime import datetime
from .get_attributes import get_attr


def calc_error_rate(input_file, output_dir):
    startTime = datetime.now()

    names = get_attr()

    # get data from analysis file
    with open(input_file, 'r', newline='') as f1:
        df = pd.read_csv(f1, header=0)
        f1.close()

    # calculate error rate and save as csv file
    with open(output_dir, 'w', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(['attribute number', 'attribute name', 'error rate in %'])
        overall = 0
        count = 0
        for attribute in names:
            # create new dataframe with only one attribute
            sub_df = df[df['attribute name'] == attribute]
            errors = sub_df['error'].sum()
            if errors == 0:
                print(f'Error rate for {attribute}: 0%')
                writer.writerow([count,
                                 attribute,
                                 0])
                count += 1
                continue
            error_rate = round(((errors/len(sub_df))*100), 2)
            print(f'Error rate for {attribute}: {error_rate}%')
            writer.writerow([count,
                             attribute,
                             error_rate])
            overall += error_rate
            count += 1
        print(f'Error rate overall: {round(overall/40, 2)}%')

    f2.close()

    print(f'The error rate has been calculated within: {datetime.now() - startTime}')
