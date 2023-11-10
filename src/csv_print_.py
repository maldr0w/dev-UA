def write_csv_data(csv_array):
    try:
        with open('democsv_.csv', 'a') as f:
            for data_point in csv_array:
                x0_, x1_ = data_point.split(',')
                x0 = x0_.replace('.', ',')
                x1 = x1_.replace('.', ',')
                f.write(f'\"{x0}\",\"{x1}\"\n')

            f.write('\n')
            f.close()
            print('Done writing to CSV file\n(Change file name or manually clear the file between runs to "freshen up"')
    except OSError as err:
        print(err)
    except ValueError as err:
        print(err)
    except Exception as err:
        print(err)
