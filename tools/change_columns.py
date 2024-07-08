import pandas as pd
# from pandas import DataFrame
import argparse
import logging
import sys


def parse_command_line_args():
    # parse command line arguments
    global g_args

    parser = argparse.ArgumentParser(description="Change columns in a csvfile")
    parser.add_argument("-i", "--input_file", nargs="?", type=argparse.FileType('r'),
                        default=sys.stdin, help="input csv file")
    parser.add_argument("-o", "--output_file", nargs="?", help="output csv file",
                        default=sys.stdout)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--remove_columns", metavar="COL_NAME", nargs="+",
                       help="column names to remove")
    group.add_argument("-k", "--keep_columns", metavar="COL_NAME", nargs="+",
                       help="columns names to keep")
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


def validate_column_names_exist(dataframe, col_names):
    col_names_in_csv = list(dataframe)
    errors = False
    for col in col_names:
        if col not in col_names_in_csv:
            print("column {} does not exist in file {}".format(col, g_args.input_file))
            errors = True
    if errors:
        exit(1)


def main():
    parse_command_line_args()

    # don't interpret empty cells or NULLs as nan, keeping foating point precision so you don't lose any digits
    dataframe1 = pd.read_csv(g_args.input_file, keep_default_na=False, float_precision='round_trip')

    columns = g_args.remove_columns
    if columns is None:
        columns = g_args.keep_columns
    validate_column_names_exist(dataframe1, columns)

    new_df = None
    if g_args.remove_columns is not None:
        # drop columns
        new_df = dataframe1.drop(columns=columns)
    elif g_args.keep_columns is not None:
        new_df = dataframe1[columns].copy()
    # write the new output file
    new_df.to_csv(g_args.output_file, index=False, na_rep='NULL')


# main
if __name__ == "__main__":
    main()
