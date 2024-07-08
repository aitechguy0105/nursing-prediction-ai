import logging
import typing

import pandas as pd

from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.common.merge_data import MergeDataBase


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MergeDataMultipleClients(MergeDataBase):
    def execute(
        self,
        *,
        run_id: str,
        client_configurations: typing.List[ClientConfiguration],
        force_regenerate: typing.Optional[bool] = False,
    ):
        dataset_provider, config = self.get_dataset_provider_and_config(run_id=run_id, force_regenerate=force_regenerate)

        self.DATASETS_TO_LOAD = config.automatic_training.features_list

        missing_datasets = []

        for feature_group in self.DATASETS_TO_LOAD:

            all_clients_data = []
            is_final_df_generated = False

            for client_configuration in client_configurations:
                client = client_configuration.client

                df, is_final_df_generated = self.load_data(
                    dataset_provider=dataset_provider,
                    client=client,
                    feature_group=feature_group,
                )

                if is_final_df_generated:
                    log.info(f"Dataset {feature_group} already exists. Skipping merging step")
                    break

                if df is not None:
                    all_clients_data.append(df)

            if is_final_df_generated:
                continue

            if len(all_clients_data) == 0:
                log.info(f"No data found for {feature_group}. Skipping merging step")
                missing_datasets.append(feature_group)
                continue

            if len(all_clients_data) != len(client_configurations):
                log.warning(f"Data for {feature_group} is missing for some clients. Merging what we have.")

            log.info(f"Merging {feature_group} data.")

            df = pd.concat(all_clients_data, ignore_index=True)

            dataset_provider.set(df=df, dataset_name=feature_group, step=self.CURRENT_STEP)

        self.upload_config(
            dataset_provider=dataset_provider,
            missing_datasets=missing_datasets,
            config=config
        )
