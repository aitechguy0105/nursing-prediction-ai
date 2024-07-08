import io
import logging
import os
import typing
import sys
from dataclasses import asdict

import fire
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image
from eliot import to_file

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.utils import load_x_y_idens_training_pipeline
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ExperimentDates
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATA_DISTRIBUTION


def get_facilities_from_data(*, df=None, categorical_features=None, pandas_categorical=None):
    if not df is None:
        return list(df.facility.unique())
    else:
        return pandas_categorical[categorical_features.index('facility')]


def get_mm_distribution(*, df, dtype, start_date, end_date, model_type):
    positive = df.query(f'target_3_day_{model_type} == 1').shape[0]
    negative = df.query(f'target_3_day_{model_type} != 1').shape[0]

    n2p = round(negative / positive, 2)
    total_patient_days = df.shape[0]
    positive_percent = round((100 * positive) / total_patient_days, 2)
    negative_percent = round((100 * negative) / total_patient_days, 2)

    return [total_patient_days, positive, negative, positive_percent, negative_percent, n2p, dtype, start_date, end_date]


def get_stay_length(*, staylength):
    if staylength > 120:
        return 120
    else:
        return staylength


def get_metrics_df(*, df, data_type, model_type):
    # here each index indicates LOS and value indicates the count of transfer for that LOS
    total = [0 for i in range(0, 121)]
    positive = [0 for i in range(0, 121)]
    negative = [0 for i in range(0, 121)]

    for index, row in df.iterrows():
        if pd.isnull(row['days_since_last_admission']):  # very rare situation but still may happen
            continue
        j = int(get_stay_length(staylength=row['days_since_last_admission']))
        total[j] += 1
        if row[f'target_3_day_{model_type}'] == 1:
            positive[j] += 1
        elif row[f'target_3_day_{model_type}'] != 1:
            negative[j] += 1

    # create a dataframe from the above 3 lists
    metric_df = pd.DataFrame({"ALL": total, "POSITIVE": positive, "NEGATIVE": negative})

    ## percentages at lengthofstay n
    metric_df['positive_percent'] = ((metric_df['POSITIVE'] / metric_df['ALL']) * 100).round(2)
    metric_df['negative_percent'] = ((metric_df['NEGATIVE'] / metric_df['ALL']) * 100).round(2)

    metric_df.columns = [data_type + '_' + col for col in metric_df.columns]
    metric_df = metric_df.fillna(0)

    return metric_df


def get_bar_graph(*, final_df, pclass, data_type, selectedClass, client, colour):
    _final_df = final_df.drop(final_df.tail(1).index)
    # _final_df = final_df

    fig = px.bar(
        _final_df,
        y=[selectedClass],
        x=list(_final_df.index),
        title=f'LOS Histogram for {pclass} patient days in {data_type} dataset for {client}',
        labels={
            'y': 'Length Of Stay',
            'caught_rth': 'LOS Count'
        },
        color_discrete_sequence=[colour]
    )
    fig['layout']['xaxis']['title'] = "Count"

    return fig


def get_line_graph(*, final_df, selectedClasses, pclass, client):
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


@convert_input_params_decorator
def data_distribution(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    model_type: typing.Optional[str] = 'MODEL_UPT',
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Generate the data distribution plots.

        :param run_id: the run id
        :param client_configurations: list of all client configurations
        :param model_type: the model type
        :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    client = "+".join([c.client for c in client_configurations])

    model_type = model_type.lower()

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f'/{model_type}')

    config = load_config(LOCAL_TRAINING_CONFIG_PATH)

    if dataset_provider.does_file_exist(
        filename=f'{model_type}/distribution_{client}',
        step=CURRENT_STEP,
        file_format="png"
    ):
        log.info(f"Distribution plot already exists for {client}_{model_type} - skipping this step")
        return

    experiment_dates = asdict(ExperimentDates(**config.training_config.training_metadata.experiment_dates.dates_calculation))

    # starting training from day 31 so that cumsum window 2,7,14,30 are all initial correct.
    experiment_dates['train_start_date'] = (pd.to_datetime(experiment_dates['train_start_date']) + pd.DateOffset(days=30)).date()

    log.info(f'MODEL: {model_type}')

    train_x, train_target_3_day, _ = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='train',
    )

    valid_x, valid_target_3_day, _ = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='valid',
    )

    test_x, test_target_3_day, _ = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='test',
    )

    np_mode = isinstance(train_x, np.ndarray)

    if np_mode:
        cate_columns = dataset_provider.load_pickle(filename=f'{model_type}/cate_columns', step=TrainingStep.DATASETS_GENERATION)
        feature_names = dataset_provider.load_pickle(filename=f'{model_type}/feature_names', step=TrainingStep.DATASETS_GENERATION)
        pandas_categorical = dataset_provider.load_pickle(filename=f'{model_type}/pandas_categorical', step=TrainingStep.DATASETS_GENERATION)

        train_x = pd.DataFrame(train_x, columns=feature_names)
        valid_x = pd.DataFrame(valid_x, columns=feature_names)
        test_x = pd.DataFrame(test_x, columns=feature_names)

    train_x[f'target_3_day_{model_type}'] = train_target_3_day
    valid_x[f'target_3_day_{model_type}'] = valid_target_3_day
    test_x[f'target_3_day_{model_type}'] = test_target_3_day

    if np_mode:
        facilities = [f.split('_')[1] for f in get_facilities_from_data(
            categorical_features=cate_columns,
            pandas_categorical=pandas_categorical
        )]
    else:
        facilities = [f.split('_')[1] for f in get_facilities_from_data(df=train_x)]

    client_df = pd.DataFrame(
        columns=["Client", client],
        data=[['Facilities', ','.join(facilities)], ['Facility count', len(facilities)]]
    )

    data_list = []
    data_list.append(
        get_mm_distribution(
            df=train_x,
            dtype='TRAIN',
            start_date=experiment_dates['train_start_date'],
            end_date=experiment_dates['train_end_date'],
            model_type=model_type
        )
    )
    data_list.append(
        get_mm_distribution(
            df=valid_x,
            dtype='VALID',
            start_date=experiment_dates['validation_start_date'],
            end_date=experiment_dates['validation_end_date'],
            model_type=model_type
        )
    )
    data_list.append(
        get_mm_distribution(
            df=test_x,
            dtype='TEST',
            start_date=experiment_dates['test_start_date'],
            end_date=experiment_dates['test_end_date'],
            model_type=model_type
        )
    )

    dist_df = pd.DataFrame(
        columns=["Patient days", "Positive", "Negative", "Positive%", "Negative%", "N2P Ratio", "TYPE", "start_date", "end_date"],
        data=data_list
    )
    dist_df['days'] = (pd.to_datetime(dist_df['end_date']) - pd.to_datetime(dist_df['start_date'])).dt.days

    _df1 = get_metrics_df(
        df=train_x[['days_since_last_admission', f'target_3_day_{model_type}']],
        data_type='TRAIN',
        model_type=model_type
    )
    _df2 = get_metrics_df(
        df=valid_x[['days_since_last_admission', f'target_3_day_{model_type}']],
        data_type='VALID',
        model_type=model_type
    )
    _df3 = get_metrics_df(
        df=test_x[['days_since_last_admission', f'target_3_day_{model_type}']],
        data_type='TEST',
        model_type=model_type
    )

    final_df = pd.concat([_df1, _df2, _df3], axis=1)

    final_df[["TRAIN_POSITIVE_nor", "VALID_POSITIVE_nor", "TEST_POSITIVE_nor"]] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        final_df[["TRAIN_POSITIVE", "VALID_POSITIVE", "TEST_POSITIVE"]]
    )

    final_df[["TRAIN_NEGATIVE_nor", "VALID_NEGATIVE_nor", "TEST_NEGATIVE_nor"]] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        final_df[["TRAIN_NEGATIVE", "VALID_NEGATIVE", "TEST_NEGATIVE"]]
    )

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=("POS Train", "POS Valid", "POS Test", "Normalised POS patient days", "NEG Train", "NEG Valid", "NEG Test", "Normalised NEG patient days")
    )

    plot1 = get_bar_graph(
        final_df=final_df,
        pclass='Positive',
        data_type='Train',
        selectedClass='TRAIN_POSITIVE',
        client=client,
        colour='blue'
    )
    plot2 = get_bar_graph(
        final_df=final_df,
        pclass='Positive',
        data_type='Valid',
        selectedClass='VALID_POSITIVE',
        client=client,
        colour='Red'
    )
    plot3 = get_bar_graph(
        final_df=final_df,
        pclass='Positive',
        data_type='Test',
        selectedClass='TEST_POSITIVE',
        client=client,
        colour='Green'
    )

    plot4 = get_bar_graph(
        final_df=final_df,
        pclass='Negative',
        data_type='Train',
        selectedClass='TRAIN_NEGATIVE',
        client=client,
        colour='blue'
    )
    plot5 = get_bar_graph(
        final_df=final_df,
        pclass='Negative',
        data_type='Valid',
        selectedClass='VALID_NEGATIVE',
        client=client,
        colour='Red'
    )
    plot6 = get_bar_graph(
        final_df=final_df,
        pclass='Negative',
        data_type='Test',
        selectedClass='TEST_NEGATIVE',
        client=client,
        colour='Green'
    )

    plot7 = get_line_graph(
        final_df=final_df,
        selectedClasses=["TRAIN_POSITIVE_nor", "VALID_POSITIVE_nor", "TEST_POSITIVE_nor"],
        pclass='Positive',
        client=client
    )
    plot8 = get_line_graph(
        final_df=final_df,
        selectedClasses=["TRAIN_NEGATIVE_nor", "VALID_NEGATIVE_nor", "TEST_NEGATIVE_nor"],
        pclass='Negative',
        client=client
    )

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

    fig.update_layout(
        height=900,
        width=1024,
        title_text=f"LOS Histogram for patient days for {client}"
    )

    pio.write_image(fig, 'distribution_plot.png')

    layout = go.Layout(
        autosize=False,
        width=1000,
        height=300
    )

    layout2 = go.Layout(
        autosize=False,
        width=1000,
        height=300,
        title="* We discard first 30 days of training data to get correct cumsum 2/7/14/30 days calculations",
    )

    fig1 = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(dist_df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=dist_df.transpose().values.tolist(),
                fill_color='lavender',
                align='left'
            )
        )
        ], layout=layout2
    )

    fig2 = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(client_df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=client_df.transpose().values.tolist(),
                fill_color='lavender',
                align='left'
            )
        )
        ], layout=layout
    )

    pio.write_image(fig1, 'distribution_table.png')
    pio.write_image(fig2, 'client_table.png')

    image0 = Image.open('./client_table.png', 'r')
    image1 = Image.open('./distribution_table.png', 'r')
    image2 = Image.open('./distribution_plot.png', 'r')

    image0 = image0.resize((image2.width, image0.height))
    image1 = image1.resize((image2.width, image1.height))
    dst = Image.new('RGB', (image1.width, image0.height + image1.height + image2.height), (250, 250, 250))
    dst.paste(image0, (0, 0))
    dst.paste(image1, (0, image0.height))
    dst.paste(image2, (0, image1.height + image0.height))
    buffer = io.BytesIO()
    dst.save(buffer, "PNG")
    dataset_provider.store_file(
        filename=f"{model_type}/distribution_{client}",
        buffer=buffer,
        step=CURRENT_STEP,
        file_format="png"
    )

    os.remove("distribution_plot.png")
    os.remove("client_table.png")
    os.remove("distribution_table.png")

    dataset_provider.store_config(step=CURRENT_STEP, prefix=f'/{model_type}')


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(data_distribution)
