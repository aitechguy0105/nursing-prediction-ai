import pandas as pd

def compare_model_performances(run1, run2, model_type='upt'):
    """
    Example Usage:
    run1 = pd.read_csv("performance_test_base.csv")
    run2 = pd.read_csv("performance_test_base2.csv")
    compare_model_performances(run1, run2)
    
    returns -{
            "common_events": common_rows, 
            "common_events_correct_predictions_a": common_a, 
            "common_events_correct_predictions_b": common_b, 
            "common_events_predicted_correctly_a_notb": common_a_minus_b,
            "common_events_predicted_correctly_a_intersection_b": common_a_intersect_b,
            "common_events_predicted_correctly_b_nota": common_b_minus_a,
            "events_other_than_common_events_correct_predictions_a": a - common_a,
            "events_other_than_common_events_correct_predictions_b": b - common_b
           }
    """
    model_type = model_type.lower()
    
    assert f'positive_date_{model_type}' in run1.columns, f"Column: positive_date_{model_type} not found in DataFrame"
    assert f'positive_date_{model_type}' in run2.columns, f"Column: positive_date_{model_type} not found in DataFrame"
    
    # getting max of 'show_in_report' for each run to see if in any of 4 opportunities we predict a transfer
    df1 = run1.groupby(['masterpatientid', 'facilityid', f'positive_date_{model_type}'])['show_in_report'].max().reset_index()
    df1.rename(columns={'show_in_report': 'show_in_report_run1'}, inplace=True)

    df2 = run2.groupby(['masterpatientid', 'facilityid', f'positive_date_{model_type}'])['show_in_report'].max().reset_index()
    df2.rename(columns={'show_in_report': 'show_in_report_run2'}, inplace=True)
    
    # merging the two dataframes
    merged_df = pd.merge(df1, df2, on=['masterpatientid', 'facilityid', f'positive_date_{model_type}'], how='outer')
    
    common_rows = merged_df.dropna(subset=['show_in_report_run1', 'show_in_report_run2']).shape[0]
    assert common_rows > 0, \
        "No common rows found in the two files"

    # calculating the metrics
    a = merged_df['show_in_report_run1'].sum()
    b = merged_df['show_in_report_run2'].sum()
    
    
    common_a = merged_df.dropna(subset=['show_in_report_run1', 'show_in_report_run2'])['show_in_report_run1'].sum()
    common_b = merged_df.dropna(subset=['show_in_report_run1', 'show_in_report_run2'])['show_in_report_run2'].sum()
    common_a_minus_b = ((merged_df['show_in_report_run1'] == True) & (merged_df['show_in_report_run2'] == False)).sum()
    common_a_intersect_b = ((merged_df['show_in_report_run1'] == True) & (merged_df['show_in_report_run2'] == True)).sum()
    common_b_minus_a = ((merged_df['show_in_report_run1'] == False) & (merged_df['show_in_report_run2'] == True)).sum()
    
    print(f"Total Common Events (in both the files): {common_rows}")
    print(f"Common Events: Predicted correctly in Run 1: {common_a}")
    print(f"Common Events: Predicted correctly in Run 2: {common_b}")
    print(f"Common Events: Predicted correctly in Run 1 but not in Run 2 (A - B): {common_a_minus_b}")
    print(f"Common Events: Predicted correctly in both Run 1 and Run 2 (A ∩ B): {common_a_intersect_b}")
    print(f"Common Events: Predicted correctly in Run 2 but not in Run 1 (B - A): {common_b_minus_a}")
    print(f"\nEvents apart from common ones predicted correctly in Run 1: {a - common_a} out of {df1.shape[0] - common_rows} events")
    print(f"Events apart from common ones predicted correctly in Run 2: {b - common_b} out of {df2.shape[0] - common_rows} events")
    return {
            "common_events": common_rows, 
            "common_events_correct_predictions_a": common_a, 
            "common_events_correct_predictions_b": common_b, 
            "common_events_predicted_correctly_a_notb": common_a_minus_b,
            "common_events_predicted_correctly_a_intersection_b": common_a_intersect_b,
            "common_events_predicted_correctly_b_nota": common_b_minus_a,
            "events_other_than_common_events_correct_predictions_a": a - common_a,
            "events_other_than_common_events_correct_predictions_b": b - common_b
           }
