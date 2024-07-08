import pandas as pd
# from pandas import DataFrame
import math
import os
import logging
import argparse


def parse_command_line_args():
    # parse command line arguments
    global g_args

    parser = argparse.ArgumentParser(description="Compare two csv files")
    parser.add_argument("-f1", "--file1", required=True, help="csv file1")
    parser.add_argument("-f2", "--file2", required=True, help="csv file2")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-c", "--columns", metavar="COL_NAME", nargs="+",
                       help="compare only these columns")
    group.add_argument("-i", "--ignore", metavar="COL_NAME", nargs="+",
                       help="ignore these columns in comparison")
    parser.add_argument("-p", "--precision", action="store_true", help="compare floats within 0.001")
    parser.add_argument("--loglevel", help="set log level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')
    parser.add_argument("--logfile", help="logfile name")
    g_args = parser.parse_args()

    log_level_map = {'DEBUG': logging.DEBUG,
                     'INFO': logging.INFO,
                     'WARNING': logging.WARNING,
                     'ERROR': logging.ERROR,
                     'CRITICAL': logging.CRITICAL}

    log_format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    if g_args.logfile:
        logging.basicConfig(filename=g_args.logfile,
                            level=log_level_map[g_args.loglevel], format=log_format)
    else:
        logging.basicConfig(level=log_level_map[g_args.loglevel], format=log_format)
    main_logger = logging.getLogger(__name__)
    main_logger.debug("Started Main")


def list_diff(li1, li2):
    return list(set(li1) - set(li2))


def compare_headers(df1, df2):
    columns_file1 = list(df1)
    columns_file2 = list(df2)
    if g_args.columns:
        set1 = set(columns_file1)
        set2 = set(columns_file2)
        column_args_set = set(g_args.columns)

        not_in_file1 = list(column_args_set - set1)
        not_in_file2 = list(column_args_set - set2)
        if not_in_file1:
            print("{} does not have the following column headers: {}".format(g_args.file1, not_in_file1))
        if not_in_file2:
            print("{} does not have the following column headers: {}".format(g_args.file2, not_in_file2))
    else:
        if g_args.ignore:
            columns_file1 = list_diff(columns_file1, g_args.ignore)
            columns_file2 = list_diff(columns_file2, g_args.ignore)
        set1 = set(columns_file1)
        set2 = set(columns_file2)

        if set1 != set2:
            print("column names don't match!")
            file1_extra = list(set1 - set2)
            file2_extra = list(set2 - set1)
            if file1_extra:
                print("{} has the following extra column headers: {}".format(g_args.file1, file1_extra))
            if file2_extra:
                print("{} has the following extra column headers: {}".format(g_args.file2, file2_extra))


def compare_row(row_num, dict_file1, dict_file2):
    logger = logging.getLogger(__name__)
    if g_args.columns:
        cols_to_compare = g_args.columns
    else:
        # compare all columns from file1
        cols_to_compare = dict_file1.keys()
        # uncomment next line to only compare Diag* columns
        # cols_to_compare = list(filter(lambda x: x.startswith("Diag"), cols_to_compare))
        if g_args.ignore:
            cols_to_compare = list_diff(cols_to_compare, g_args.ignore)
    for key1, value1 in dict_file1.items():
        logger.debug("key1={}, value1={}".format(key1, value1))
        if key1 in cols_to_compare and key1 in dict_file2.keys():
            value2 = dict_file2[key1]
            logger.debug("row_num:{}, comparing key: {}, value1={}, value2={}".format(row_num, key1, value1, value2))
            values_equal = True
            if isinstance(value1, float) and math.isnan(value1):
                if isinstance(value2, float):
                    if not math.isnan(value2):
                        values_equal = False
                else:
                    values_equal = False
            elif g_args.precision and isinstance(value1, float):
                if not isinstance(value2, float):
                    values_equal = False
                elif abs(value2 - value1) > 0.001:
                    values_equal = False
            elif value1 != value2:
                values_equal = False
            if not values_equal:
                print("FileName={};RowNum={};ColumnName={};Value={}".format(g_args.file1, row_num, key1, value1))
                print("FileName={};RowNum={};ColumnName={};Value={}".format(g_args.file2, row_num, key1, value2))
                print("")
    return


def compare(df1, df2):
    # first compare column names
    compare_headers(df1, df2)
    list_of_dict1 = df1.to_dict('records')
    list_of_dict2 = df2.to_dict('records')

    len1 = len(list_of_dict1)
    len2 = len(list_of_dict2)
    if len1 != len2:
        print("{} has {} lines while {} has {} lines!".format(g_args.file1, len1, g_args.file2, len2))
    index = 0
    while index < len1 and index < len2:
        dict_file1 = list_of_dict1[index]
        dict_file2 = list_of_dict2[index]
        compare_row(index, dict_file1, dict_file2)
        index += 1


def main():
    parse_command_line_args()

    if not os.path.isfile(g_args.file1) or not os.access(g_args.file2, os.R_OK):
        print("File {} is not present or is not readable".format(g_args.file1))
        exit(1)

    if not os.path.isfile(g_args.file2) or not os.access(g_args.file2, os.R_OK):
        print("File {} is not present or is not readable".format(g_args.file2))
        exit(1)

    if g_args.columns is not None:
        print("only comparing the following columns: {}".format(g_args.columns))

    if g_args.ignore is not None:
        print("ignoring the following columns in comparison: {}".format(g_args.ignore))

    if g_args.precision:
        print("comparing floats within precision of 0.001")

    # don't interpret empty cells or NULLs as nan
    dataframe1 = pd.read_csv(g_args.file1, keep_default_na=False, float_precision='round_trip', dtype='str')
    dataframe2 = pd.read_csv(g_args.file2, keep_default_na=False, float_precision='round_trip', dtype='str')
    compare(dataframe1, dataframe2)


# main
if __name__ == "__main__":
    main()
