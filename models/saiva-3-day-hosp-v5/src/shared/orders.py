import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class OrderFeatures(BaseFeaturizer):
    def __init__(self, census_df, orders, training=False):
        self.census_df = census_df
        self.orders = orders
        self.training = training
        super(OrderFeatures, self).__init__()

    def get_diagnostic_orders(self):
        diagnostic_orders = self.orders.loc[(self.orders.ordercategory == 'Diagnostic') & (~self.orders.orderdescription.isna())].copy()
        diagnostic_orders['orderdate'] = diagnostic_orders.orderdate.dt.date
        diagnostic_orders['count_indicator_diagnostic_orders'] = 1

        diagnostic_pivoted = diagnostic_orders.drop(
            columns=['patientid', 'ordercategory', 'ordertype', 'orderdescription', 'pharmacymedicationname',
                     'diettype', 'diettexture', 'dietsupplement']).pivot_table(
            index=['masterpatientid', 'facilityid', 'orderdate'], values=['count_indicator_diagnostic_orders'],
            aggfunc=sum
        ).reset_index()

        diagnostic_pivoted['orderdate'] = pd.to_datetime(diagnostic_pivoted.orderdate)
        diagnostic_pivoted.columns = 'order_' + diagnostic_pivoted.columns
        del diagnostic_orders
        return diagnostic_pivoted

    def get_diettype_diettexture_orders(self):
        diet_orders = self.orders[self.orders.ordercategory == 'Dietary - Diet'].copy()
        diet_orders['orderdate'] = diet_orders.orderdate.dt.date
        diet_orders['indicator'] = 1
        diet_orders = diet_orders.drop_duplicates(
            subset=['masterpatientid', 'orderdate', 'diettype', 'diettexture']
        )

        diet_type_pivoted = diet_orders.loc[:, ['masterpatientid', 'orderdate', 'diettype', 'indicator']].pivot_table(
            index=['masterpatientid', 'orderdate'],
            columns=['diettype'],
            values='indicator',
            aggfunc=min
        ).reset_index()

        diet_type_pivoted.head()
        diet_type_pivoted['orderdate'] = pd.to_datetime(diet_type_pivoted.orderdate)
        diet_type_pivoted.columns = 'order_' + diet_type_pivoted.columns

        diet_texture_pivoted = diet_orders.loc[:,
                               ['masterpatientid', 'orderdate', 'diettexture', 'indicator']].pivot_table(
            index=['masterpatientid', 'orderdate'],
            columns=['diettexture'],
            values='indicator',
            aggfunc=min
        ).reset_index()

        diet_texture_pivoted['orderdate'] = pd.to_datetime(diet_texture_pivoted.orderdate)
        diet_texture_pivoted.columns = 'order_' + diet_texture_pivoted.columns

        return diet_type_pivoted, diet_texture_pivoted

    def get_dietsupplement_orders(self):
        diet_supplements = self.orders[self.orders.ordercategory == 'Dietary - Supplements'].copy()
        if len(diet_supplements):
            diet_supplements['orderdate'] = diet_supplements.orderdate.dt.date
            diet_supplements['indicator'] = 1
            diet_supplements = diet_supplements.drop_duplicates(
                subset=['masterpatientid', 'orderdate', 'dietsupplement'])

            diet_supplements_pivoted = diet_supplements.loc[:,
                                       ['masterpatientid', 'orderdate', 'dietsupplement', 'indicator']].pivot_table(
                index=['masterpatientid', 'orderdate'],
                columns='dietsupplement',
                values='indicator',
                aggfunc=min).reset_index()

            diet_supplements_pivoted['orderdate'] = pd.to_datetime(diet_supplements_pivoted.orderdate)

            # =========================dietsupplement count column added========================
            diet_supplements_counts = diet_supplements.groupby(
                ['masterpatientid', 'facilityid', 'orderdate']).dietsupplement.count().reset_index().rename(
                columns={'dietsupplement': 'count_indicator_dietsupplement'})

            diet_supplements_counts['orderdate'] = pd.to_datetime(diet_supplements_counts.orderdate)
            diet_supplements_pivoted.columns = 'order_' + diet_supplements_pivoted.columns
            diet_supplements_counts.columns = 'order_' + diet_supplements_counts.columns

            return diet_supplements_pivoted, diet_supplements_counts
        else:
            return pd.DataFrame(), pd.DataFrame()

    def generate_features(self):
        """
        - diagnostic_orders_count & dietsupplement_count columns added
        - diettype, diettexture, dietsupplement values are made as separate columns and added to parent df
        """
        log_message(message_type='info', message='Orders Processing...')
        self.orders = self.sorter_and_deduper(
            self.orders,
            sort_keys=['masterpatientid', 'orderdate'],
            unique_keys=['masterpatientid', 'orderdate', 'orderdescription']
        )
        # =========================Count of diagnostic_orders========================

        diagnostic_df = self.get_diagnostic_orders()
        base1 = self.census_df.merge(
            diagnostic_df,
            how='left',
            left_on=['masterpatientid', 'facilityid', 'censusdate'],
            right_on=['order_masterpatientid', 'order_facilityid', 'order_orderdate']
        )

        del self.census_df
        del diagnostic_df

        # =========================diettype columns added========================
        diet_type_df, diet_texture_df = self.get_diettype_diettexture_orders()
        base2 = base1.merge(
            diet_type_df,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['order_masterpatientid', 'order_orderdate']
        )

        del base1
        del diet_type_df
        base3 = base2.merge(
            diet_texture_df,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['order_masterpatientid', 'order_orderdate']
        )

        del base2
        del diet_texture_df

        # =========================dietsupplement columns added========================
        diet_supplements_df, diet_supplements_count_df = self.get_dietsupplement_orders()

        if diet_supplements_df.empty and diet_supplements_count_df.empty:
            base5 = base3.copy()
        else:
            base3 = self.drop_columns(base3, '_x$|_y$')
            base4 = base3.merge(
                diet_supplements_df,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['order_masterpatientid', 'order_orderdate']
            )
            base5 = base4.merge(
                diet_supplements_count_df,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['order_masterpatientid', 'order_orderdate']
            )

            del diet_supplements_df
            del base4
            del diet_supplements_count_df

        del base3
        # drop any duplicated columns
        base5 = base5.loc[:, ~base5.columns.duplicated()]

        # drop unwanted columns
        base5 = self.drop_columns(
            base5,
            'date_of_transfer|_masterpatientid|_facilityid|orderdate|createddate|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )

        # =============Trigger garbage collection & downcast to free-up memory ==================
        gc.collect()
        base5 = self.downcast_dtype(base5)

        assert base5.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')
        base5 = self.add_na_indicators(base5, self.ignore_columns)
        cols = [col for col in base5.columns if col.startswith('order')]
        # Do cumulative summation on all order columns
        log_message(message_type='info', message='cumulative summation...')
        base5 = self.get_cumsum_features(cols, base5)

        return base5
