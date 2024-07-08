import gc
import sys
import functools
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action

from .featurizer import BaseFeaturizer

class OrderFeatures(BaseFeaturizer):

    def __init__(self, census_df, orders, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.orders = orders
        self.training = training
        self.config = config
        super(OrderFeatures, self).__init__()

    @staticmethod
    def merge_dfs(dfs_list):
        # merge the dataframes on the specified keys
        merged_df = functools.reduce(
            lambda left, right: pd.merge(left, right, on=['masterpatientid', 'censusdate'], how='outer')
            , dfs_list)
        return merged_df

    # Diagnostic, Enteral - Feed, and Laboratory all use the same logic and can be done at the same time
    def conditional_get_order_category_features(self):
        cat_names = ['Diagnostic', 'Enteral - Feed', 'Laboratory']
        cat_orders = self.orders.loc[
            (self.orders.ordercategory.isin(cat_names)) & (~self.orders.orderdescription.isna())
        ].copy().assign(
            ordercategory=lambda x: x['ordercategory'].map({
                'Diagnostic': 'diagnostic_count_indicator_diagnostic_orders',
                'Enteral - Feed': 'enteral_count_indicator_enteral_orders',
                'Laboratory': 'laboratory_count_indicator_laboratory_orders',
            })
        ).drop(columns=['patientid', 'ordertype', 'orderdescription', 'pharmacymedicationname',
                    'diettype', 'diettexture', 'dietsupplement'
        ])

        return self.conditional_featurizer_all(cat_orders, 'ordercategory', prefix='order')
    
    def conditional_get_stat_orders(self):
        stat_orders = self.orders.copy()
        stat_orders = stat_orders[stat_orders['orderdescription'].str.contains('STAT', regex=False, na=False)]
        stat_orders = stat_orders.drop(
            columns=['patientid', 'ordercategory', 'ordertype', 'orderdescription', 'pharmacymedicationname',
                     'diettype', 'diettexture', 'dietsupplement'])
        stat_orders['feature_name'] = 'count_indicator_stat_orders'

        return self.conditional_featurizer_all(stat_orders, 'feature_name', prefix='order_stat')
    
    def conditional_get_other_orders(self):
        other_orders = self.orders[self.orders.ordercategory == 'Other'].copy().drop(
            columns=['patientid', 'ordercategory', 'orderdescription', 'pharmacymedicationname',
                     'diettype', 'diettexture', 'dietsupplement'])
        other_orders['ordertype'] = other_orders['ordertype'].str.lower()

        return self.conditional_featurizer_all(other_orders, 'ordertype', prefix='order_other') 
    

    def conditional_get_diettype_diettexture_orders(self):
        diet_orders = self.orders[self.orders.ordercategory == 'Dietary - Diet'].copy()
        diet_orders['fluidconsistency'] = diet_orders['fluidconsistency'].fillna('Missing')

        diet_orders = diet_orders.drop_duplicates(
            subset=['masterpatientid', 'orderdate', 'diettype', 'diettexture', 'fluidconsistency']
        )
        diet_orders['diettype'] = diet_orders['diettype'].str.lower()
        diet_orders['diettexture'] = diet_orders['diettexture'].str.lower()
        diet_orders['fluidconsistency'] = diet_orders['fluidconsistency'].str.lower()

        days_last_event_df1, cuminx_df1, cumsum_df1 = self.conditional_featurizer_all(diet_orders, 'diettype', prefix='order_diettype')

        days_last_event_df2, cuminx_df2, cumsum_df2 = self.conditional_featurizer_all(diet_orders, 'diettexture', prefix='order_diettexture')

        days_last_event_df3, cuminx_df3, cumsum_df3 = self.conditional_featurizer_all(diet_orders, 'fluidconsistency', prefix='order_fluidconsistency')

        return self.merge_dfs([days_last_event_df1, days_last_event_df2, days_last_event_df3]), \
            self.merge_dfs([cuminx_df1, cuminx_df2, cuminx_df3]), \
            self.merge_dfs([cumsum_df1, cumsum_df2, cumsum_df3])

    def conditional_get_dietsupplement_orders(self):
        diet_supplements = self.orders[self.orders.ordercategory == 'Dietary - Supplements'].copy()
        if diet_supplements.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
        diet_supplements = diet_supplements.drop_duplicates(
        subset=['masterpatientid', 'orderdate', 'dietsupplement'])[['masterpatientid', 'orderdate', 'ordercreateddate', 'dietsupplement']]
        diet_supplements['dietsupplement'] = diet_supplements['dietsupplement'].str.lower()

        days_last_event_df1, cuminx_df1, cumsum_df1 = self.conditional_featurizer_all(diet_supplements, 'dietsupplement', prefix='order_supplements')

        diet_supplements = diet_supplements[diet_supplements.dietsupplement.notnull()]
        diet_supplements['dietsupplement'] = 'count_indicator_dietsupplement'
        days_last_event_df2, cuminx_df2, cumsum_df2 = self.conditional_featurizer_all(diet_supplements, 'dietsupplement', prefix='order_supplements')

        return self.merge_dfs([days_last_event_df1, days_last_event_df2]), \
            self.merge_dfs([cuminx_df1, cuminx_df2]), \
            self.merge_dfs([cumsum_df1, cumsum_df2])     

    def get_diagnostic_orders(self):
        diagnostic_orders = self.orders.loc[(self.orders.ordercategory == 'Diagnostic') & (~self.orders.orderdescription.isna())].copy()
        diagnostic_orders['column_name'] = 'count_indicator_diagnostic_orders'
        diagnostic_pivoted = self.pivot_patient_date_event_count(diagnostic_orders, groupby_column='column_name', 
                                    date_column='orderdate', prefix='order_diagnostic')

        del diagnostic_orders
        return diagnostic_pivoted

    def get_enteral_orders(self):
        enteral_orders = self.orders.loc[(self.orders.ordercategory == 'Enteral - Feed') & (~self.orders.orderdescription.isna())].copy()
        enteral_orders['column_name'] = 'count_indicator_enteral_orders'
        enteral_pivoted = self.pivot_patient_date_event_count(enteral_orders, groupby_column='column_name', date_column='orderdate', prefix='order_enteral')

        del enteral_orders
        return enteral_pivoted
    
    
    def get_stat_orders(self):        
        stat_orders = self.orders.loc[self.orders['orderdescription'].str.contains('STAT', regex=False, na=False)].copy()
        stat_orders['column_name'] = 'count_indicator_stat_orders'
        stat_orders_pivoted = self.pivot_patient_date_event_count(stat_orders, groupby_column='column_name', date_column='orderdate', prefix='order_stat')

        del stat_orders
        return stat_orders_pivoted

    def get_laboratory_orders(self):
        laboratory_orders = self.orders.loc[(self.orders.ordercategory == 'Laboratory') & (~self.orders.orderdescription.isna())].copy()        
        laboratory_orders['column_name'] = 'count_indicator_laboratory_orders'
        laboratory_pivoted = self.pivot_patient_date_event_count(laboratory_orders, groupby_column='column_name', date_column='orderdate', prefix='order_laboratory')

        del laboratory_orders
        return laboratory_pivoted
    
    def get_other_orders(self):
        other_orders = self.orders[self.orders.ordercategory == 'Other'].copy()
        other_orders['ordertype'] = other_orders['ordertype'].str.lower()        
        
        if self.config.featurization.orders.pivot_aggfunc_sum:
            aggfunc='sum' 
        else:
            aggfunc='min'
        other_orders_pivoted = self.pivot_patient_date_event_count(other_orders, groupby_column='ordertype', 
                    date_column='orderdate', prefix='order_other', aggfunc=aggfunc)
        
        return other_orders_pivoted

    def get_diettype_diettexture_orders(self):
        diet_orders = self.orders[self.orders.ordercategory == 'Dietary - Diet'].copy()
        diet_orders['fluidconsistency'] = diet_orders['fluidconsistency'].fillna('Missing')
        # dropping duplicates will often prevent counting greater than 1 for each day, and we already drop duplicate orders in sorter_and_deduper
        # diet_orders = diet_orders.drop_duplicates(
        #     subset=['masterpatientid', 'orderdate', 'diettype', 'diettexture', 'fluidconsistency']
        # )
        diet_orders['diettype'] = diet_orders['diettype'].str.lower()
        diet_orders['diettexture'] = diet_orders['diettexture'].str.lower()
        diet_orders['fluidconsistency'] = diet_orders['fluidconsistency'].str.lower()

        if self.config.featurization.orders.pivot_aggfunc_sum:
            aggfunc='sum' 
        else:
            aggfunc='min'

        diet_type_pivoted = self.pivot_patient_date_event_count(diet_orders, groupby_column='diettype', 
                                date_column='orderdate', prefix='order_diettype', aggfunc=aggfunc)
        diet_texture_pivoted = self.pivot_patient_date_event_count(diet_orders, groupby_column='diettexture', 
                                date_column='orderdate', prefix='order_diettexture', aggfunc=aggfunc)
        fluidconsistency_pivoted = self.pivot_patient_date_event_count(diet_orders, groupby_column='fluidconsistency', 
                                date_column='orderdate', prefix='order_fluidconsistency', aggfunc=aggfunc)
        
        return diet_type_pivoted, diet_texture_pivoted, fluidconsistency_pivoted


    def get_dietsupplement_orders(self):
        diet_supplements = self.orders[self.orders.ordercategory == 'Dietary - Supplements'].copy()
        if not diet_supplements.empty:
            # diet_supplements = diet_supplements.drop_duplicates(
            #     subset=['masterpatientid', 'orderdate', 'dietsupplement'])
            diet_supplements['dietsupplement'] = diet_supplements['dietsupplement'].str.lower()

            if self.config.featurization.orders.pivot_aggfunc_sum:
                aggfunc='sum' 
            else:
                aggfunc='min'
            
            diet_supplements_pivoted = self.pivot_patient_date_event_count(diet_supplements, groupby_column='dietsupplement', 
                            date_column='orderdate', prefix='order_supplements', aggfunc=aggfunc)

            # =========================dietsupplement count column added========================
            diet_supplements['orderdate'] = diet_supplements['orderdate'].dt.normalize()

            diet_supplements_counts = diet_supplements.groupby(
                ['masterpatientid', 'orderdate']).dietsupplement.count().reset_index().rename(
                columns={'dietsupplement': 'count_indicator_dietsupplement'})
            
            diet_supplements_counts.columns = ['masterpatientid', 'censusdate'] + \
                        ['order_supplements_' + name for name in diet_supplements_counts.columns[2:]]

            return diet_supplements_pivoted, diet_supplements_counts
        else:
            return pd.DataFrame(), pd.DataFrame()

    def generate_features(self):
        """
        - diagnostic_orders_count & dietsupplement_count columns added
        - diettype, diettexture, dietsupplement values are made as separate columns and added to parent df
        """
        with start_action(action_type=f"Orders - generating orders features", orders_shape= self.orders.shape):
            start = time.time()
            self.orders = self.sorter_and_deduper(
                self.orders,
                sort_keys=['masterpatientid', 'orderdate'],
                unique_keys=['masterpatientid', 'orderdate', 'ordercategory', 'ordertype', 'orderdescription',
                             'diettype', 'diettexture', 'dietsupplement', 'fluidconsistency']
            )
            # ========================= diagnostic_orders conditioned on orderdate <= createddate ========================
            log_message(message_type='info', 
                        message=f'Orders - use_conditional_functions -> {self.config.featurization.orders.use_conditional_functions}'
                       )
            if self.config.featurization.orders.use_conditional_functions:
                if self.config.featurization.orders.generate_na_indicators:
                    raise NotImplementedError("generate_na_indicators not yet implemented for conditional_functions.")
                # conditional orders
                log_message(message_type='info', message='Orders - executing conditional orders features.')

                # results from each function will be a tuple of 3 dataframes corresponding to days_since, cumidx, and cumsum features
                feature_tuples = []
                log_message(message_type='info', message='Orders - executing conditional diet type and diet texture orders features.')
                feature_tuples.append(self.conditional_get_diettype_diettexture_orders())
                log_message(message_type='info', message='Orders - executing conditional diet supplement orders features.')
                feature_tuples.append(self.conditional_get_dietsupplement_orders())
                log_message(message_type='info', message='Orders - executing conditional other orders features.')
                feature_tuples.append(self.conditional_get_other_orders())
                log_message(message_type='info', message='Orders - executing conditional stat orders features.')
                feature_tuples.append(self.conditional_get_stat_orders())
                log_message(message_type='info', message='Orders - executing conditional diet category orders features.')
                feature_tuples.append(self.conditional_get_order_category_features()) # conditional_get_order_category_features is equivalent to running get_diagnostic_orders, get_enteral_orders, and get_laboratory_orders


                days_last_event_df = self.merge_dfs([feature_dfs[0] for feature_dfs in feature_tuples])
                log_message(message_type='info', 
                        message=f'Orders - created features to count days since last order event. -> {days_last_event_df.shape}'
                       )
                cumidx_df = self.merge_dfs([feature_dfs[1] for feature_dfs in feature_tuples])
                log_message(message_type='info', 
                        message=f'Orders - cumulative summation, patient days with any events cumsum. -> {cumidx_df.shape}'
                       )
                cumsum_df = self.merge_dfs([feature_dfs[2] for feature_dfs in feature_tuples])
                log_message(message_type='info', 
                        message=f'Orders - cumulative summation, total number of events cumsum. -> {cumsum_df.shape}'
                       )
                final = self.census_df.merge(cumidx_df, on=['masterpatientid','censusdate'])
                final = final.merge(cumsum_df, on=['masterpatientid','censusdate'])
                final = final.merge(days_last_event_df , on=['masterpatientid','censusdate'])

                for feature_tuple in feature_tuples:
                    for df in feature_tuple:
                        del df

            else:
                diagnostic_df = self.get_diagnostic_orders()
                generate_na_indicators = self.config.featurization.orders.generate_na_indicators
                base = self.census_df.merge(
                    diagnostic_df,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )

                del diagnostic_df

                order_features_list = []
                # base = self.census_df

                # order_features_list.append(self.get_diagnostic_orders())
                log_message(message_type='info', message='Orders - executing diet type and diet texture orders features.')
                order_features_list.extend(list(self.get_diettype_diettexture_orders()))
                log_message(message_type='info', message='Orders - executing diet supplement orders features.')
                order_features_list.extend(list(self.get_dietsupplement_orders()))
                log_message(message_type='info', message='Orders - executing other orders features.')
                order_features_list.append(self.get_other_orders())
                log_message(message_type='info', message='Orders - executing enteral orders features.')
                order_features_list.append(self.get_enteral_orders())
                log_message(message_type='info', message='Orders - executing stat orders features.')
                order_features_list.append(self.get_stat_orders())
                log_message(message_type='info', message='Orders - executing laboratory orders features.')
                order_features_list.append(self.get_laboratory_orders())

                while order_features_list:
                    order_feature_df = order_features_list.pop(0)
                    if order_feature_df.shape[0]>0:
                        base = base.merge(
                        order_feature_df,
                        how='outer',
                        on=['masterpatientid', 'censusdate']
                        )

                # drop any duplicated columns
                assert base.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

                # drop unwanted columns
                base = self.drop_columns(
                    base,
                    '_masterpatientid|_facilityid|orderdate|createddate'
                )

                # =============Trigger garbage collection & downcast to free-up memory ==================
                gc.collect()
                base = self.downcast_dtype(base)

                base = base.drop_duplicates()
                assert base.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

                # handle NaN by adding na indicators


                if generate_na_indicators:
                    log_message(message_type='info', message='Orders - creating na indicators and handling na values...')
                    base = self.add_na_indicators(base, self.ignore_columns)
                cols = [col for col in base.columns if col.startswith('order')]

                #########################################################################################
                events_df = base[['facilityid','masterpatientid','censusdate']+cols]
                events_df[cols] = events_df[cols].fillna(0)
                events_df = self.downcast_dtype(events_df)

                # count days since last events
                log_message(message_type='info', message=f'Orders - creating days since last event feature.')
                days_last_event_df = self.apply_n_days_last_event(events_df, cols)
                ##########################################################################################     

                # Do cumulative summation on all order columns
                cumidx_df = self.get_cumsum_features(cols, base, cumidx=True)
                log_message(
                    message_type='info', 
                    message=f'Orders - cumulative summation, patient days with any events cumsum.', 
                    cumidx_df_shape = cumidx_df.shape
                )
                cumsum_df = self.get_cumsum_features(cols, events_df, cumidx=False)
                log_message(
                    message_type='info', 
                    message=f'Orders - cumulative summation, total number of events cumsum.',
                    cumsum_df_shape = cumsum_df.shape
                )
                final = self.census_df.merge(cumidx_df, on=['facilityid','masterpatientid','censusdate'])
                final = final.merge(cumsum_df, on=['facilityid','masterpatientid','censusdate'])
                final = final.merge(days_last_event_df , on=['masterpatientid','censusdate'])

                del events_df, base

            self.sanity_check(final)
            del cumidx_df, cumsum_df
            del days_last_event_df 
            log_message(
                message_type='info', 
                message=f'Orders - exiting orders', 
                final_dataframe_shape=final.shape,
                time_taken=round(time.time() - start, 2)
            )
            return final
