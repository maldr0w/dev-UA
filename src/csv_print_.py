from pathlib import Path
CSV_FOLDER = Path("csv/")
def write_csv_data(f_name: str, csv_array):
    f_path = CSV_FOLDER / f"{f_name}.csv"
    try:
        with open(f_path, 'a') as f:
            for data_point in csv_array:
                x0_, x1_ = data_point.split(',')
                x0 = x0_.replace('.', ',')
                x1 = x1_.replace('.', ',')
                f.write(f'\"{x0}\",\"{x1}\"\n')

            f.write('\n')
            f.close()
            print('Done writing to CSV file\n')
    except OSError as err:
        print(err)
    except ValueError as err:
        print(err)
    except Exception as err:
        print(err)
