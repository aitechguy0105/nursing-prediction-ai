import os
from svglib.svglib import SvgRenderer, svg2rlg
from reportlab.graphics import renderPDF
from PyPDF2 import PdfReader, PdfMerger
import gc
import glob
import json
import numpy as np
import pandas as pd
import swifter
from pathlib import Path
# pandas_profiling has been deprecated in the report
#from pandas_profiling import ProfileReport
import warnings
from more_itertools import chunked_even
import plotly
import pickle
from IPython.display import HTML
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from scipy.stats import kstest, spearmanr, kendalltau
from sklearn.metrics import matthews_corrcoef
from PIL import Image
from tqdm import tqdm
from math import ceil
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

pd.set_option('display.max_colwidth', 0)
tqdm.pandas()


class Datacard:
    """

    The Datacard Class. This class runs the X,Y datacard,
    detailed in JIRA SAIV-2284.
    To interact with the class, construct it as follows:

#    >>>> DC = Datacard(client='Avante',
        #               data_paths=data_paths,
        #               report_conf_dir=conf_dir,
        #               report_output_dir=output_dir,
        #               experiment_dates=EXPERIMENT_DATES)

    and then run the .generate_datacard() method.
#    >>>> DC.generate_datacard()

    """
    def __init__(self,
                 report_conf_dir,
                 report_output_dir,
                 data_paths,
                 client,
                 experiment_dates,
                 tests={'COL': {'largest_is_norm': {'name': 'is_norm', 'direction': 'largest',
                                     'title': 'This table shows the numeric columns which have been found to be normally distributed',
                                     'col_widths': [3, 1, 3, 1, 3, 1]},
                 'largest_most_common_value_in_col': {'name': 'most_common_value_in_col', 'most_common_value_in_col': 'most_common_pct_in_col', 'direction': 'largest',
                                                      'title': 'This table shows the most common nonzero element in a specific column',
                                                      'col_widths': [4, 2, 1, 4, 2, 1, 4, 2, 1]},
                 'largest_outlier_pct': {'name': 'outlier_pct', 'direction': 'largest',
                                         'title': 'This table shows the numeric columns with most outliers and their percentages (by IQR)',
                                         'col_widths': [3, 1, 3, 1, 3, 1]},
                 'largest_nans_in_col': {'name': 'nans_in_col', 'direction': 'largest',
                                         'title': 'This table shows the columns with the most NaNs, and their percentages',
                                         'col_widths': [3, 1, 3, 1, 3, 1]},
                 'largest_zeros_in_col': {'name': 'zeros_in_col','direction': 'largest',
                                          'title': 'This table shows the columns with the most zeros, and their percentages',
                                          'col_widths': [3, 1, 3, 1, 3, 1]},
                 'largest_skew': {'name': 'skew', 'direction': 'largest',
                                  'title': 'This table shows the numeric columns which are the most skewed',
                                  'col_widths': [3, 1, 3, 1, 3, 1]}},
        'ROW': {'largest_most_common_value_in_row': {'name': 'most_common_value_in_row', 'most_common_value_in_row': 'most_common_pct_in_row', 'direction': 'largest',
                                                     'title': 'This table shows the most common nonzero element in a specific row',
                                                     'col_widths': None},
                'largest_nans_in_row': {'name': 'nans_in_row', 'direction': 'largest',
                                        'title': 'This table shows the rows with the most NaNs, and their percentages',
                                                'col_widths': None},
                'largest_zeros_in_row': {'name': 'zeros_in_row', 'direction': 'largest',
                                         'title': 'This table shows the rows with the most zeros, and their percentages',
                                                'col_widths': None}},
        'MULTICOL': {'largest_corr_to_target': {'name': 'corr_type', 'corr_type': 'corr_value', 'direction': 'largest',
                                                'title': 'This table shows the columns with the highest correlation to the target, and the type of correlation',
                                                'col_widths': [4, 2, 1, 4, 2, 1, 4, 2, 1]},
                     'smallest_corr_to_target': {'name': 'corr_type', 'corr_type': 'corr_value', 'direction': 'smallest',
                                                 'title': 'This table shows the columns with the lowest correlation to the target, and the type of correlation',
                                                 'col_widths': [4, 2, 1, 4, 2, 1, 4, 2, 1]}},
        'MULTIROW': {},
        'IDENS': {}},
                 layouts={'general': go.Layout(
                                autosize=False,
                                width=2500,
                                height=1000,
                                font={'size': 32},
                            ),
                                'dist': go.Layout(
                                autosize=False,
                                width=2500,
                                height=500,
                                font={'size': 32},
                                title="* We discard first 30 days of training data to get correct cumsum 2/7/14/30 days calculations",
                            ),
                                'client': go.Layout(
                                autosize=False,
                                width=2500,
                                height=360,
                                font={'size': 32}
                            ),
                                'alerts': go.Layout(
                                autosize=False,
                                font={'size': 32},
                                width=2500)},
                 admissions_col='days_since_last_admission',
                 idens_pos_col='positive_date_upt',
                 idens_los_col='hosp_lengthofstay',
                 idens_incident_data_col='incidentdate_upt',
                 feat_groups=['demo', 'vtl', 'census', 'order', 'admissions', 'med', 'dx', 'alert'],
                 alert_feat_groups=['demo', 'vtl', 'census', 'order', 'admissions', 'med', 'dx', 'alert'],
                 label_col='hosp_target_3_day_hosp',
                 top_n=10,
                 day_thresh=120):

        self.data_paths = data_paths
        self.layouts = layouts
        self.tests = tests
        self.client = client
        self.experiment_dates = experiment_dates
        self.admissions_col = admissions_col
        self.idens_pos_col = idens_pos_col
        self.idens_los_col = idens_los_col
        self.idens_incident_data_col = idens_incident_data_col
        self.report_conf_dir = report_conf_dir
        self.report_output_dir = report_output_dir
        self.label_col = label_col
        # TODO: Accept only either feat_groups or grouped_feats
        self.feat_groups = feat_groups
        self.alert_feat_groups = alert_feat_groups
        self.top_n = top_n
        self.day_thresh = day_thresh
        self.data = {}

    def generate_datacard(self):

        # generate report dir, if it doesn't exist
        if not os.path.exists(self.report_output_dir):
            os.makedirs(self.report_output_dir)

        # load X, Y matrices
        print('Loading Dataset (pickle files)')
        for dataset_name, dataset in self.data_paths.items():
            self.data.update({dataset_name: self._read_pickle(dataset)})

        ########################## Datacard V1 plots ####################
        print('Generating Client Table')
        # client table
        self._add_client_basics_to_report(train_x=self.data['X_train'])

        print('Generating Dataset Highlights')
        # Dataset highlights + idens comparison
        dist_df, final_df = self._add_dataset_highlights_to_report()

        print('Generating LOS Histogram')
        # LOS Histogram
        self._add_LOS_histogram_to_report(final_df=final_df)

        df_train = self._merge_X_Y_data('train')
        df_valid = self._merge_X_Y_data('valid')
        df_test = self._merge_X_Y_data('test')

        del self.data['X_train']
        del self.data['X_valid']
        del self.data['X_test']

        gc.collect()
        # keep only relevant columns (defined by FEAT_GROUPS)

        df_train = self._keep_only_requested_cols(df=df_train, keep_cols=self.feat_groups + [self.label_col])
        df_valid = self._keep_only_requested_cols(df=df_valid, keep_cols=self.feat_groups + [self.label_col])
        df_test = self._keep_only_requested_cols(df=df_test, keep_cols=self.feat_groups + [self.label_col])

        # #######################
        # # SINGLE COLUMN TESTS #
        # #######################
        print('Checking single column tests')
        self._run_single_column_tests_wrapper(df_train=df_train,
                                              df_valid=df_valid,
                                              df_test=df_test)

        #####################
        ## SINGLE ROW TESTS #
        #####################
        print('Checking single row tests')
        self._run_single_row_tests_wrapper(df_train=df_train,
                                           df_valid=df_valid,
                                           df_test=df_test)

        # ############################
        # # MULTI-COLUMN TESTS #
        # ############################
        print('Checking multicol tests')
        self._run_col_vs_col_tests_wrapper(df_train=df_train,
                                           df_valid=df_valid,
                                           df_test=df_test)

        # ############################
        # # MULTI-ROW TESTS #
        # ############################
        print('Checking multirow tests')
        self._run_row_vs_row_tests_wrapper(df_train=df_train,
                                           df_valid=df_valid,
                                           df_test=df_test)

        ####################
        # IDENS ANALYSIS  ##
        ####################
        print('running idens analysis')
        self._run_idens_analysis_wrapper(df_train_idens=self.data['idens_train'],
                                         df_valid_idens=self.data['idens_valid'],
                                         df_test_idens=self.data['idens_test'])

        ##########################
        ### Pandas Profiling #####
        ##########################

        # print('Running Pandas Profiling - This Could Take a While!')
        # self._add_insights_from_profiling(profiling_dict={'Train': df_train, 'Valid': df_valid, 'Test': df_test})

        ###############################
        # Combine figures for report ##
        ###############################

        # # TODO: Update this
        # self._combine_figures_for_report(figures=['basic_stats-client_table.svg',
        #                                           'basic_stats-distribution_table.svg',
        #                                           'basic_stats-distribution_plot.svg',
        #                                           'idens_test-num_events.svg',
        #                                           'col_test-largest_is_norm_table.svg',
        #                                           'col_test-largest_most_common_value_in_col_table.svg',
        #                                           'col_test-largest_nans_in_col_table.svg',
        #                                           'col_test-largest_outlier_pct_table.svg',
        #                                           'col_test-largest_skew_table.svg',
        #                                           'col_test-largest_zeros_in_col_table.svg',
        #                                           'col_vs_col_test-largest_corr_type_table.svg',
        #                                           'col_vs_col_test-smallest_corr_type_table.svg',
        #                                           'row_test-largest_most_common_value_in_row_table.svg',
        #                                           'row_test-largest_nans_in_row_table.svg',
        #                                           'row_test-largest_zeros_in_row_table.svg'])

        figures = ['basic_stats-client_table.svg','basic_stats-distribution_table.svg',
                   'basic_stats-distribution_plot.svg','idens_test-num_events.svg','col_test-largest_is_norm_table.svg',
                   'col_test-largest_most_common_value_in_col_table.svg','col_test-largest_nans_in_col_table.svg',
                   'col_test-largest_outlier_pct_table.svg', 'col_test-largest_skew_table.svg',
                   'col_test-largest_zeros_in_col_table.svg', 'col_vs_col_test-largest_corr_type_table.svg',
                   'col_vs_col_test-smallest_corr_type_table.svg', 'row_test-largest_most_common_value_in_row_table.svg',
                   'row_test-largest_nans_in_row_table.svg', 'row_test-largest_zeros_in_row_table.svg']

        figures = [os.path.join(self.report_output_dir, fig) for fig in figures]
        self._combine_figures_for_report_svg_pdf(figures=figures)

        print('Done!')

    ################################################################

    def _add_insights_from_profiling(self, profiling_dict):
        splits = list(profiling_dict.keys())
        insights = {}
        split_alerts = {}

        # # run pandas profiling on each dataset and save it (6:30 hours of runtime, val/test report)
        for split, df in profiling_dict.items():
            # run pandas profiling if report doesn't exist
            html_path = os.path.join(self.report_output_dir, f'datacard_{split}.html')
            json_path = os.path.join(self.report_output_dir, f'datacard_{split}.json')

            if (not os.path.isfile(html_path)) or (not os.path.isfile(json_path)):
                profile = ProfileReport(df, config_file=self.report_conf_dir / 'minimal_conf.yaml')
                profile.to_file(html_path)
                profile.to_file(json_path)

            else:
                print(f'profiling report found, skipping')
            # TODO: Continue here after Reports have been generated

            with open(json_path) as json_file:
                report_data = json.load(json_file)
            #
            # consider additional insights here
            # alert insights if they belong to selected feat groups
            alerts = [i for e in self.alert_feat_groups for i in report_data['alerts'] if e in i]
            alerts.sort()
            split_alerts.update({'alerts': alerts})
            insights.update({split: split_alerts})

        # add additional insights here
        # TODO: choose multiindex hierarchy (relevant when we'll want more insights)

        alerts_df = pd.concat([pd.DataFrame.from_dict(insights[split], orient='index').T for split in splits], axis=1)
        alerts_df.columns = splits

        # save alerts / insights:
        # set height of Alerts table dynamically with how many alerts there are
        if len(alerts_df) > 0:
            self.layouts['alerts']['height'] = 25 * len(alerts_df)

            alerts_df_fig = go.Figure(data=[go.Table(
                header=dict(values=list(alerts_df.columns),
                            align='center'),
                cells=dict(values=alerts_df.transpose().values.tolist(),
                           align='center'))
            ], layout=self.layouts['alerts'])

            pio.write_image(alerts_df_fig, 'alerts_table.svg')
        else:
            print(f'no alerts found for {self.alert_feat_groups}!')

    @staticmethod
    def _intersect(l1, l2):
        return [value for value in l1 if value in l2]

    @staticmethod
    def _get_facilities_from_data(df):
        return list(df.facility.unique())

    def _get_mm_distribution(self, df, split, start_date, end_date):
        positive = df.query(f'{self.label_col} == 1').shape[0]
        negative = df.query(f'{self.label_col} != 1').shape[0]

        n2p = round(negative / positive, 3)
        total_patient_days = df.shape[0]
        positive_percent = (100 * positive) / total_patient_days
        negative_percent = (100 * negative) / total_patient_days

        return [total_patient_days, positive, negative, positive_percent, negative_percent, n2p, split, start_date,
                end_date]

    @staticmethod
    def _get_stay_length(staylength, day_thresh):
        # TODO: Change this to determined dynamically
        if staylength > day_thresh:
            return day_thresh
        else:
            return staylength

    def _get_metrics_df(self, df, split):
        # here each index indicates LOS and value indicates the count of transfer for that LOS
        total = [0 for i in range(0, 121)]
        positive = [0 for i in range(0, 121)]
        negative = [0 for i in range(0, 121)]

        for index, row in df.iterrows():
            j = int(self._get_stay_length(row[self.admissions_col], day_thresh=self.day_thresh))
            total[j] += 1
            if row[f'{self.label_col}'] == 1:
                positive[j] += 1
            elif row[f'{self.label_col}'] != 1:
                negative[j] += 1

        # create a dataframe from the above 3 lists
        metric_df = pd.DataFrame({"ALL": total, "POSITIVE": positive, "NEGATIVE": negative})

        ## percentages at lengthofstay n
        metric_df['positive_percent'] = (metric_df['POSITIVE'] / metric_df['ALL']) * 100
        metric_df['negative_percent'] = (metric_df['NEGATIVE'] / metric_df['ALL']) * 100

        metric_df.columns = [split + '_' + col for col in metric_df.columns]
        metric_df = metric_df.fillna(0)

        return metric_df

    @staticmethod
    def _is_categorical(array_like):
        return array_like.dtype.name == 'category'

    @staticmethod
    def _get_bar_graph(final_df, pclass, split, selectedClass, client, colour):
        _final_df = final_df.drop(final_df.tail(1).index)
        # _final_df = final_df

        fig = px.bar(
            _final_df,
            y=[selectedClass],
            x=list(_final_df.index),
            title=f'LOS Histogram for {pclass} patient days in {split} dataset for {client}',
            labels={
                'y': 'Length Of Stay',
                'caught_rth': 'LOS Count'
            },
            color_discrete_sequence=[colour]
        )
        fig['layout']['xaxis']['title'] = "Count"

        return fig

    @staticmethod
    def _get_line_graph(final_df, selectedClasses, pclass, client):
        _final_df = final_df.copy()
        _final_df = final_df.drop(final_df.tail(1).index)

        fig = px.line(
            _final_df,
            y=selectedClasses,
            x=list(_final_df.index),
            labels={
                'x': 'Length Of Stay',
            },
            title=f'LOS Histogram for Normalised {pclass} patient days across Train, Valid & Test dataset for {client}',
        )
        fig['layout']['yaxis']['title'] = f"{pclass} Patient day Normalised value between 0 to 100"

        return fig

    @staticmethod
    def _read_pickle(path):
        with open(path, 'rb') as f:
            content = pickle.load(f)
        return content

    @staticmethod
    def _plot_colored_table(combined_df, layout, output_path, col_widths=None):
        combined_df_outliers_fig = go.Figure(data=[go.Table(
            columnwidth=col_widths,
            header=dict(values=list(combined_df.columns),
                        align='center',
                        height=ceil(layout['font']['size'] * 1.2)),
            cells=dict(values=combined_df.transpose().values.tolist(),
                       align='center',
                       height=ceil(layout['font']['size'] * 1.2)))
        ], layout=layout)

        pio.write_image(combined_df_outliers_fig, output_path)

    @staticmethod
    def _build_split_output_table(stat_test_df, split, axis,
                                  series_names, top_n, direction):
        # align columns and rows

        report_header = 'feature' if axis == 'columns' else 'row'

        if axis == 'columns':
            stat_test_df = stat_test_df.T

        # single non-feature series for table
        if isinstance(series_names, str):

            if direction == 'largest':
                df_return = pd.DataFrame(pd.to_numeric(stat_test_df[series_names]).nlargest(top_n)).reset_index()

            elif direction == 'smallest':
                df_return = pd.DataFrame(pd.to_numeric(stat_test_df[series_names]).nsmallest(top_n)).reset_index()

            df_return.columns = [report_header] + [series_names]
            multi_index = pd.MultiIndex.from_product([[split], [report_header] + [series_names]])
            df_return.columns = multi_index

        # multiple non-feature series (dataframe) for table [dict]. Select over value and display both.
        # i.e. {'most_common_value_in_col': 'most_common_pct_in_col'}
        elif isinstance(series_names, dict):
            multicols = list(series_names.keys()) + list(series_names.values())
            sub_df = stat_test_df[multicols]
            if direction == 'largest':
                df_return = sub_df.apply(pd.to_numeric, errors='ignore').nlargest(top_n, columns=list(
                    series_names.values())).reset_index()

            elif direction == 'smallest':
                df_return = sub_df.apply(pd.to_numeric, errors='ignore').nsmallest(top_n, columns=list(
                    series_names.values())).reset_index()

            df_return.columns = [report_header] + multicols
            multi_index = pd.MultiIndex.from_product([[split], [report_header] + multicols])
            df_return.columns = multi_index

        return df_return

    def _run_single_column_tests_wrapper(self, df_train, df_valid, df_test):
        train_single_column_tests_df = df_train.swifter.apply(self._run_single_column_tests, axis=0)
        valid_single_column_tests_df = df_valid.swifter.apply(self._run_single_column_tests, axis=0)
        test_single_column_tests_df = df_test.swifter.apply(self._run_single_column_tests, axis=0)

        for _, test_content in self.tests['COL'].items():
            # TODO: not a great determinant, but if 'name' key exists as a value, it's a multicolumn situation
            intersection = self._intersect(l1=list(test_content.keys()),
                                           l2=list(test_content.values()))
            if intersection:
                series_name = {intersection[0]: test_content[intersection[0]]}
            else:
                series_name = test_content['name']

            train_df = self._build_split_output_table(stat_test_df=train_single_column_tests_df,
                                                      split='Train',
                                                      axis='columns',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            valid_df = self._build_split_output_table(stat_test_df=valid_single_column_tests_df,
                                                      split='Valid',
                                                      axis='columns',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            test_df = self._build_split_output_table(stat_test_df=test_single_column_tests_df,
                                                     split='Test',
                                                     axis='columns',
                                                     series_names=series_name,
                                                     top_n=self.top_n,
                                                     direction=test_content['direction'])

            combined_df = pd.concat([train_df, valid_df, test_df], axis=1)
            combined_df = combined_df.round(2)

            if isinstance(series_name, dict):
                stat_table_name = list(series_name.keys())[0]
            else:
                stat_table_name = series_name

            # modify test layout here
            TEST_LAYOUT = self.layouts['general']
            TEST_LAYOUT['title'] = test_content['title']
            self._plot_colored_table(combined_df=combined_df,
                                     layout=TEST_LAYOUT,
                                     col_widths=test_content['col_widths'],
                                     output_path=f"col_test-{test_content['direction']}_{stat_table_name}_table.svg")

    def _run_single_column_tests(self, col):
        # TODO: ADD MORE COLUMN-SPECIFIC TESTS HERE

        # check normality
        is_norm = np.nan
        outlier_pct = np.nan
        skew = np.nan
        most_common_value_in_col = np.nan
        most_common_pct_in_col = np.nan
        nans_in_col = np.nan
        zeros_in_col = np.nan

        col_no_na = col.dropna()

        if len(col_no_na) > 0:

            nans_in_col = self._calc_nan_pct(ser=col)
            zeros_in_col = self._calc_zeros_pct(ser=col)

            # get most common non-zero value for columns with not only zeros
            zeros_mask = (col_no_na == 0)
            if not zeros_mask.all():
                most_common_value_in_col, most_common_pct_in_col = self._get_most_common_element(
                    ser=col_no_na[~zeros_mask])

            if self._is_categorical(col):
                pass

            # only relevant for categorical datatypes (imbalance)
            elif col.dtype == 'object':
                pass

            # if only relevant for numeric datatypes
            elif np.issubdtype(col.dtype, np.number):
                is_norm = self._check_normality(ser=col_no_na)

                # check outliers (IQR Method [we can do +/- 3std for normal distribution])
                outlier_pct = self._calc_outliers_pct(ser=col_no_na)

                # check skewed distribution
                skew = self._calc_skew(ser=col_no_na)

        return pd.Series([is_norm, outlier_pct,
                          nans_in_col, zeros_in_col,
                          most_common_value_in_col,
                          most_common_pct_in_col,
                          skew],
                         index=['is_norm', 'outlier_pct',
                                'nans_in_col', 'zeros_in_col',
                                'most_common_value_in_col', 'most_common_pct_in_col',
                                'skew'],
                         name=col.name)

    @staticmethod
    def _check_normality(ser):
        statistic, pval = kstest(ser, 'norm')
        return pval >= 0.05

    def _calc_outliers_pct(self, ser):
        # IQR method
        return self._outliers_iqr(ser)

    @staticmethod
    def _outliers_iqr(ser):
        quartile_1, quartile_3 = np.percentile(ser, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return (sum((ser > upper_bound) | (ser < lower_bound)) / len(ser)) * 100

    def _run_single_row_tests_wrapper(self, df_train, df_valid, df_test):
        ddf_train = dd.from_pandas(df_train, npartitions=os.cpu_count()-1)
        with ProgressBar():
            train_single_row_tests_df = ddf_train.map_partitions(lambda df: df.apply(self._run_single_row_tests, axis=1)).compute()

        ddf_valid = dd.from_pandas(df_valid, npartitions=os.cpu_count()-1)
        with ProgressBar():
            valid_single_row_tests_df = ddf_valid.map_partitions(lambda df: df.apply(self._run_single_row_tests, axis=1)).compute()

        ddf_test = dd.from_pandas(df_test, npartitions=os.cpu_count()-1)
        with ProgressBar():
            test_single_row_tests_df = ddf_test.map_partitions(lambda df: df.apply(self._run_single_row_tests, axis=1)).compute()

        for _, test_content in self.tests['ROW'].items():

            intersection = self._intersect(l1=list(test_content.keys()),
                                           l2=list(test_content.values()))
            if intersection:
                series_name = {intersection[0]: test_content[intersection[0]]}
            else:
                series_name = test_content['name']

            train_df = self._build_split_output_table(stat_test_df=train_single_row_tests_df,
                                                      split='Train',
                                                      axis='rows',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            valid_df = self._build_split_output_table(stat_test_df=valid_single_row_tests_df,
                                                      split='Valid',
                                                      axis='rows',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            test_df = self._build_split_output_table(stat_test_df=test_single_row_tests_df,
                                                     split='Test',
                                                     axis='rows',
                                                     series_names=series_name,
                                                     top_n=self.top_n,
                                                     direction=test_content['direction'])

            combined_df = pd.concat([train_df, valid_df, test_df], axis=1)
            combined_df = combined_df.round(2)

            if isinstance(series_name, dict):
                stat_table_name = list(series_name.keys())[0]
            else:
                stat_table_name = series_name

            # modify test layout here
            TEST_LAYOUT = self.layouts['general']
            TEST_LAYOUT['title'] = test_content['title']

            # TEST_LAYOUT['title'] = f'Row Test: {direction} {stat_table_name} values'
            self._plot_colored_table(combined_df=combined_df,
                                     layout=TEST_LAYOUT,
                                     col_widths=test_content['col_widths'],
                                     output_path=f"row_test-{test_content['direction']}_{stat_table_name}_table.svg")

    def _run_single_row_tests(self, row):
        # too many nans in row
        row_no_na = row.dropna()

        nans_in_row = self._calc_nan_pct(ser=row)
        zeros_in_row = self._calc_zeros_pct(ser=row)

        zeros_mask = (row_no_na == 0)
        if not zeros_mask.all():
            most_common_value_in_row, most_common_pct_in_row = self._get_most_common_element(ser=row_no_na[~zeros_mask])

        return pd.Series([nans_in_row, zeros_in_row,
                          most_common_value_in_row, most_common_pct_in_row],
                         index=['nans_in_row', 'zeros_in_row',
                                'most_common_value_in_row', 'most_common_pct_in_row'],
                         name=row.name)

    @staticmethod
    def _calc_nan_pct(ser):
        # check pct of nans in series
        return ser.isna().sum() / len(ser)

    @staticmethod
    def _calc_zeros_pct(ser):
        # check pct of zeros in series
        return (sum(ser == 0) / len(ser)) * 100

    @staticmethod
    def _get_most_common_element(ser):
        # returns the most common element in a series and its proportion
        most_common_element = ser.value_counts().index[0]
        return most_common_element, ((ser == most_common_element).sum() / len(ser)) * 100

    @staticmethod
    def _calc_skew(ser):
        return ser.skew()

    def _run_col_vs_col_tests_wrapper(self, df_train, df_valid, df_test):
        train_multicol_tests_df = df_train.loc[:, df_train.columns != self.label_col]. \
            swifter.apply(self._run_col_vs_col_tests, target_col=df_train[self.label_col], axis=0)

        valid_multicol_tests_df = df_valid.loc[:, df_valid.columns != self.label_col]. \
            swifter.apply(self._run_col_vs_col_tests, target_col=df_valid[self.label_col], axis=0)

        test_multicol_tests_df = df_test.loc[:, df_test.columns != self.label_col]. \
            swifter.apply(self._run_col_vs_col_tests, target_col=df_test[self.label_col], axis=0)

        for _, test_content in self.tests['MULTICOL'].items():

            intersection = self._intersect(l1=list(test_content.keys()),
                                           l2=list(test_content.values()))
            if intersection:
                series_name = {intersection[0]: test_content[intersection[0]]}
            else:
                series_name = test_content['name']

            train_df = self._build_split_output_table(stat_test_df=train_multicol_tests_df,
                                                      split='Train',
                                                      axis='columns',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            valid_df = self._build_split_output_table(stat_test_df=valid_multicol_tests_df,
                                                      split='Valid',
                                                      axis='columns',
                                                      series_names=series_name,
                                                      top_n=self.top_n,
                                                      direction=test_content['direction'])

            test_df = self._build_split_output_table(stat_test_df=test_multicol_tests_df,
                                                     split='Test',
                                                     axis='columns',
                                                     series_names=series_name,
                                                     top_n=self.top_n,
                                                     direction=test_content['direction'])

            combined_df = pd.concat([train_df, valid_df, test_df], axis=1)
            combined_df = combined_df.round(2)

            if isinstance(series_name, dict):
                stat_table_name = list(series_name.keys())[0]
            else:
                stat_table_name = series_name

            # modify test layout here
            TEST_LAYOUT = self.layouts['general']
            TEST_LAYOUT['title'] = test_content['title']
            self._plot_colored_table(combined_df=combined_df,
                                     layout=TEST_LAYOUT,
                                     col_widths=test_content['col_widths'],
                                     output_path=f"col_vs_col_test-{test_content['direction']}_{stat_table_name}_table.svg")

    def _run_col_vs_col_tests(self, rolling_col, target_col):
        # TODO: ADD MORE MULTI-COL-SPECIFIC TESTS HERE

        corr_type = np.nan
        corr_value = np.nan

        local_df = pd.concat([rolling_col, target_col], axis=1)

        # drop rows in which either the rolling_col or the target_col has null
        local_df.dropna(how='any', axis=0, inplace=True)

        # split the cols again
        rolling_col_no_na = local_df.iloc[:, :1].squeeze()
        target_col_no_na = local_df.iloc[:, 1:].squeeze()

        if (len(rolling_col_no_na) > 0) and (len(target_col_no_na) > 0):
            corr_type, corr_value = self._calc_correlation(rolling_col=rolling_col_no_na,
                                                           target_col=target_col_no_na)

        return pd.Series([corr_type, corr_value], index=['corr_type', 'corr_value'],
                         name=rolling_col.name)

    def _run_row_vs_row_tests_wrapper(self, df_train, df_valid, df_test):
        # TODO: ADD MORE MULTI-ROw-SPECIFIC TESTS HERE (EUCLEDIAN/COSINE DISTANCE)?
        pass

    def _run_row_vs_row_tests(self, row1, row2):
        # TODO: ADD MORE MULTI-ROw-SPECIFIC TESTS HERE (EUCLEDIAN/COSINE DISTANCE)?
        pass

    def _merge_X_Y_data(self, split):
        return pd.concat([self.data[f'X_{split}'], pd.Series(data=self.data[f'Y_{split}'], name=f'{self.label_col}')],
                         axis=1)

    def _add_client_basics_to_report(self, train_x):
        facilities = self._get_facilities_from_data(df=train_x)
        client_df = pd.DataFrame(
            columns=["Client", self.client],
            data=[['Facilities', ','.join(facilities)],
                  ['Facility count', len(facilities)]]
        )

        header = list(client_df.columns)
        cells = client_df.transpose().values.tolist()

        client_df_fig = go.Figure(data=[go.Table(
            columnwidth=[1, 4],
            header=dict(values=header,
                        align='center',
                        height=ceil(self.layouts['client']['font']['size']*1.2)),
            cells=dict(values=cells,
                       align='center',
                       height=ceil(self.layouts['client']['font']['size']*1.2)))
        ], layout=self.layouts['client'])

        pio.write_image(client_df_fig, os.path.join(self.report_output_dir, 'basic_stats-client_table.svg'))

    def _add_dataset_highlights_to_report(self):
        # Todo: get Experiment Dates

        df_train = self._merge_X_Y_data('train')
        df_valid = self._merge_X_Y_data('valid')
        df_test = self._merge_X_Y_data('test')

        data_list = []
        data_list.append(
            self._get_mm_distribution(df=df_train,
                                      split='TRAIN',
                                      start_date=self.experiment_dates['train_start_date'],
                                      end_date=self.experiment_dates['train_end_date']))
        data_list.append(self._get_mm_distribution(df=df_valid,
                                                   split='VALID',
                                                   start_date=self.experiment_dates['validation_start_date'],
                                                   end_date=self.experiment_dates['validation_end_date']))
        data_list.append(self._get_mm_distribution(df=df_test,
                                                   split='TEST',
                                                   start_date=self.experiment_dates['test_start_date'],
                                                   end_date=self.experiment_dates['test_end_date']))

        dist_df = pd.DataFrame(
            columns=["Patient days", "Positive", "Negative",
                     "Positive%", "Negative%", "N2P Ratio", "TYPE",
                     "start_date", "end_date"],
            data=data_list
        )
        dist_df['days'] = (pd.to_datetime(dist_df['end_date']) - pd.to_datetime(dist_df['start_date'])).dt.days

        _df1 = self._get_metrics_df(df=df_train[[self.admissions_col, self.label_col]], split='TRAIN')
        _df2 = self._get_metrics_df(df=df_valid[[self.admissions_col, self.label_col]], split='VALID')
        _df3 = self._get_metrics_df(df=df_test[[self.admissions_col, self.label_col]], split='TEST')

        final_df = pd.concat([_df1, _df2, _df3], axis=1)

        final_df[["TRAIN_POSITIVE_nor", "VALID_POSITIVE_nor", "TEST_POSITIVE_nor"]] = MinMaxScaler(
            feature_range=(0, 100)).fit_transform(
            final_df[["TRAIN_POSITIVE", "VALID_POSITIVE", "TEST_POSITIVE"]]
        )

        final_df[["TRAIN_NEGATIVE_nor", "VALID_NEGATIVE_nor", "TEST_NEGATIVE_nor"]] = MinMaxScaler(
            feature_range=(0, 100)).fit_transform(
            final_df[["TRAIN_NEGATIVE", "VALID_NEGATIVE", "TEST_NEGATIVE"]]
        )

        idens_vs_label_df = self._idens_vs_label_test(df_train=df_train,
                                                      df_train_idens=self.data['idens_train'],
                                                      df_valid=df_valid,
                                                      df_valid_idens=self.data['idens_valid'],
                                                      df_test=df_test,
                                                      df_test_idens=self.data['idens_test'])

        # keep only 2 digits after '.'
        dist_df = dist_df.round(2)
        idens_vs_label_df = idens_vs_label_df.round(2)

        dist_df_list = dist_df.transpose().values.tolist()
        dist_df_list.append([item[0] for item in list(idens_vs_label_df.values)])

        dist_df_fig = go.Figure(data=[go.Table(
            header=dict(values=list(dist_df.columns) + ['Census/Label Agreement %'],
                        align='center',
                        height=ceil(self.layouts['dist']['font']['size'] * 1.2)),
            cells=dict(values=dist_df_list,
                       align='center',
                       height=ceil(self.layouts['dist']['font']['size'] * 1.2)))
        ], layout=self.layouts['dist'])

        pio.write_image(dist_df_fig, os.path.join(self.report_output_dir, 'basic_stats-distribution_table.svg'))

        return dist_df, final_df

    def _add_LOS_histogram_to_report(self, final_df):
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=("POS Train", "POS Valid", "POS Test", "Normalised POS patient days",
                            "NEG Train", "NEG Valid", "NEG Test", "Normalised NEG patient days")
        )

        plot1 = self._get_bar_graph(final_df, 'Positive', 'Train', 'TRAIN_POSITIVE', self.client, 'blue')
        plot2 = self._get_bar_graph(final_df, 'Positive', 'Valid', 'VALID_POSITIVE', self.client, 'Red')
        plot3 = self._get_bar_graph(final_df, 'Positive', 'Test', 'TEST_POSITIVE', self.client, 'Green')

        plot4 = self._get_bar_graph(final_df, 'Negative', 'Train', 'TRAIN_NEGATIVE', self.client, 'blue')
        plot5 = self._get_bar_graph(final_df, 'Negative', 'Valid', 'VALID_NEGATIVE', self.client, 'Red')
        plot6 = self._get_bar_graph(final_df, 'Negative', 'Test', 'TEST_NEGATIVE', self.client, 'Green')

        plot7 = self._get_line_graph(final_df, ["TRAIN_POSITIVE_nor", "VALID_POSITIVE_nor", "TEST_POSITIVE_nor"],
                                     'Positive',
                                     self.client)
        plot8 = self._get_line_graph(final_df, ["TRAIN_NEGATIVE_nor", "VALID_NEGATIVE_nor", "TEST_NEGATIVE_nor"],
                                     'Negative',
                                     self.client)

        fig.add_trace(
            plot1["data"][0],
            row=1, col=1
        )

        fig.add_trace(
            plot2["data"][0],
            row=1, col=2
        )

        fig.add_trace(
            plot3["data"][0],
            row=1, col=3
        )

        fig.add_trace(
            plot7["data"][0],
            row=1, col=4
        )
        fig.add_trace(
            plot7["data"][1],
            row=1, col=4
        )
        fig.add_trace(
            plot7["data"][2],
            row=1, col=4
        )

        fig.add_trace(
            plot4["data"][0],
            row=2, col=1
        )

        fig.add_trace(
            plot5["data"][0],
            row=2, col=2
        )

        fig.add_trace(
            plot6["data"][0],
            row=2, col=3
        )

        fig.add_trace(
            plot8["data"][0],
            row=2, col=4
        )
        fig.add_trace(
            plot8["data"][1],
            row=2, col=4
        )
        fig.add_trace(
            plot8["data"][2],
            row=2, col=4
        )

        fig.update_layout(height=900,
                          width=1024,
                          title_text=f"LOS Histogram for patient days for {self.client}")

        pio.write_image(fig, os.path.join(self.report_output_dir,'basic_stats-distribution_plot.svg'))

    def _calc_correlation(self, rolling_col, target_col):
        # compute the relevant correlation, depending on DType
        # if only relevant for numeric datatypes

        corr_type = np.nan
        statistic = np.nan

        rolling_col_is_num = False
        target_col_is_num = np.issubdtype(target_col.dtype, np.number)

        rolling_col_is_bool = rolling_col.dtype == 'bool'
        target_col_is_bool = target_col.dtype == 'bool'

        rolling_col_is_obj = rolling_col.dtype == 'object'
        target_col_is_obj = target_col.dtype == 'object'

        rolling_col_is_cat = self._is_categorical(rolling_col)
        target_col_is_cat = self._is_categorical(target_col)

        # TODO: Fix this patch - isinstance fails for 'category types'
        if not rolling_col_is_cat:
            rolling_col_is_num = np.issubdtype(rolling_col.dtype, np.number)

        # Spearman
        if (rolling_col_is_num) and (target_col_is_num):
            # TODO: Maybe support pearson's for two normal-distributions
            corr_type = 'spearman'
            res = spearmanr(rolling_col, target_col)
            statistic, pvalue = res.correlation, res.pvalue

        elif rolling_col_is_bool and target_col_is_bool:
            corr_type = 'tetrachoric'
            statistic = matthews_corrcoef(rolling_col, target_col)


        # elif ((rolling_col_is_num) and (target_col is_cat)) or ((rolling_col_is_cat) and (target_col is_num)):
        #     pass

        # TODO: can do anova, but meh

        # kendal-tau
        elif rolling_col_is_obj and target_col_is_obj:
            corr_type = 'kendalltau'
            statistic, pvalue = kendalltau(rolling_col, target_col)

        return corr_type, statistic

    @staticmethod
    def _combine_figures_for_report(figures):
        # assume that the figures are in the correct order!
        images = [Image.open(figure) for figure in figures]

        max_width = max([image.size[0] for image in images])
        # change component size while keeping aspect ratio
        size_factor = [max_width / image.size[0] for image in images]
        images = [image.resize((int(image.size[0] * size_factor[iImage]),
                                int(image.size[1] * size_factor[iImage]))) for iImage, image in enumerate(images)]

        sum_height = sum([image.size[1] for image in images])
        dst = Image.new('RGB', (max_width, sum_height), (250, 250, 250))

        cum_height = 0
        for iImage, image in enumerate(images):
            dst.paste(image, (0, cum_height))
            cum_height += image.size[1]

        dst.save("data_card_report.svg", "SVG")

    @staticmethod
    def _resize_image(img):
        # Calculate the aspect ratio of the image and the standard page
        img_ratio = img.size[0] / img.size[1]
        page_ratio = 11 / 8.5

        # If the image is wider than the page, fit the width and scale the height
        if img_ratio > page_ratio:
            new_width = 11 * 300  # 300 DPI
            new_height = int(new_width / img_ratio)
        # Otherwise, fit the height and scale the width
        else:
            new_height = 8.5 * 300  # 300 DPI
            new_width = int(new_height * img_ratio)

        # Resize the image to fit within the page while maintaining its aspect ratio
        img = img.resize((int(new_width), int(new_height)), resample=Image.LANCZOS)
        return img

    def _combine_figures_for_report_pdf(self, figures):
        # assume that the figures are in the correct order!
        images = [Image.open(figure) for figure in figures]

        # change component size to fit to page dimensions
        images = [self._resize_image(image) for image in images]
        max_width = max([image.size[0] for image in images])
        max_height = max([image.size[1] for image in images])

        sum_height = sum([image.size[1] for image in images])
        dst = Image.new('RGB', (max_width, sum_height), (250, 250, 250))

        cum_height = 0
        for iImage, image in enumerate(images):
            dst.paste(image, (0, cum_height))
            cum_height += image.size[1]

        dst.save(os.path.join(self.report_output_dir, "data_card_report.png"), "PNG")
        dst.save(os.path.join(self.report_output_dir, "data_card_report.svg"), "SVG")

    def _combine_figures_for_report_svg_pdf(self, figures):
        # assume that the figures are in the correct order!
        # create a list of all the SVG files in the directory

        # create a PdfMerger object
        pdf_merger = PdfMerger()

        # iterate over the SVG files and convert each one to PDF
        for svg_file in figures:
            drawing = svg2rlg(svg_file)
            pdf_file = os.path.splitext(svg_file)[0] + ".pdf"
            renderPDF.drawToFile(drawing, pdf_file)

            # add the PDF file to the PdfMerger object
            with open(pdf_file, 'rb') as f:
                pdf_merger.append(PdfReader(f), import_outline=False)

        # merge all of the PDF files into a single PDF
        with open(os.path.join(self.report_output_dir, "data_card.pdf"), 'wb') as f:
            pdf_merger.write(f)

        # delete the temporary PDF files
        for svg_file in figures:
            pdf_file = os.path.splitext(svg_file)[0] + ".pdf"
            os.remove(pdf_file)

    @staticmethod
    def _keep_only_requested_cols(df, keep_cols):
        return df[[col for col in df.columns if any(sub in col for sub in keep_cols)]]

    def _idens_vs_label_test(self, df_train, df_valid, df_test, df_train_idens, df_valid_idens, df_test_idens):

        # test if label matches idens file (and, by extension, the X matrix)
        train_match = df_train[self.label_col] == ~df_train_idens[self.idens_pos_col].reset_index(drop=True).isna()
        valid_match = df_valid[self.label_col] == ~df_valid_idens[self.idens_pos_col].reset_index(drop=True).isna()
        test_match = df_test[self.label_col] == ~df_test_idens[self.idens_pos_col].reset_index(drop=True).isna()

        idens_vs_label_df = pd.DataFrame(data=[(train_match.sum() / len(train_match)) * 100,
                                               (valid_match.sum() / len(valid_match)) * 100,
                                               (test_match.sum() / len(test_match)) * 100],
                                         index=['Train', 'Valid', 'Test'],
                                         columns=['Census/Label Agreement %'])
        return idens_vs_label_df

    def _run_idens_analysis_wrapper(self, df_train_idens, df_valid_idens, df_test_idens, LOS_thresh=30):

        train_events = self._idens_event_count(idens_df=df_train_idens, LOS_thresh=LOS_thresh, split='Train')
        valid_events = self._idens_event_count(idens_df=df_valid_idens, LOS_thresh=LOS_thresh, split='Valid')
        test_events = self._idens_event_count(idens_df=df_test_idens, LOS_thresh=LOS_thresh, split='Test')

        combined_df = pd.concat([train_events, valid_events, test_events], axis=1)

        combined_df = combined_df.round(2)

        # modify test layout here
        TEST_LAYOUT = self.layouts['general']
        TEST_LAYOUT['title'] = f'Idens Test: number of events in dataset'
        self._plot_colored_table(combined_df=combined_df,
                                 layout=TEST_LAYOUT,
                                 col_widths=None,
                                 output_path=os.path.join(self.report_output_dir, 'idens_test-num_events.svg')),

    def _idens_event_count(self, idens_df, LOS_thresh, split):
        # drop los < 0:
        idens_df = idens_df[idens_df[self.idens_los_col] > 0]

        # less than group
        idens_df_short_stay = idens_df[idens_df[self.idens_los_col] <= LOS_thresh]

        # greater than group:
        idens_df_long_stay = idens_df[idens_df[self.idens_los_col] > LOS_thresh]
        mask = ~idens_df[self.idens_incident_data_col].isna()
        num_events_short = len(idens_df_short_stay[mask][self.idens_incident_data_col].unique())
        num_events_long = len(idens_df_long_stay[mask][self.idens_incident_data_col].unique())

        df_return = pd.DataFrame({'num_events_short': num_events_short, 'num_events_long': num_events_long}, index=[0])
        multi_index = pd.MultiIndex.from_product([[split], list(df_return.columns)])
        df_return.columns = multi_index

        return df_return


####################################################################


if __name__ == '__main__':
    processed_path = Path('v3_data/')
    conf_dir = Path('report_confs/')
    output_dir = Path('reports/')

    EXPERIMENT_DATES = {'train_start_date': '2020-07-01',
                        'train_end_date': '2022-01-22',
                        'validation_start_date': '2022-01-23',
                        'validation_end_date': '2022-04-27',
                        'test_start_date': '2022-04-28',
                        'test_end_date': '2022-07-31'}

    data_paths = {'X_train': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_x_upt.pickle'),
                  'Y_train': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_target_3_day_upt.pickle'),
                  'idens_train': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_idens_upt.pickle'),
                  'X_valid': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_x_upt.pickle'),
                  'Y_valid': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_target_3_day_upt.pickle'),
                  'idens_valid': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-valid_idens_upt.pickle'),
                  'X_test': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-test_x_upt.pickle'),
                  'Y_test': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-test_target_3_day_upt.pickle'),
                  'idens_test': Path('/Users/dschmidt/GitHub/tmp/v6_boris_data/final-test_idens_upt.pickle'),
    }

    DC = Datacard(data_paths=data_paths,
                  report_conf_dir=conf_dir,
                  report_output_dir=output_dir,
                  experiment_dates=EXPERIMENT_DATES,
                  client='Avante',
                  admissions_col='days_since_last_admission',
                  idens_pos_col='positive_date_upt',
                  idens_los_col='hosp_lengthofstay',
                  idens_incident_data_col='incidentdate_upt',
                  feat_groups=['demo', 'vtl', 'census', 'order', 'admissions', 'med', 'dx', 'alert'],
                  alert_feat_groups=['demo', 'vtl', 'census', 'order', 'admissions', 'med', 'dx', 'alert'],
                  label_col='hosp_target_3_day_hosp',
                  top_n=10,
                  day_thresh=120)

    DC.generate_datacard()
