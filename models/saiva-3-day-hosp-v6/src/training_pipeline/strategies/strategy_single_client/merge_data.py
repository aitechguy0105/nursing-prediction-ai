import logging
import typing

from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.common.merge_data import MergeDataBase


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MergeDataSingleClient(MergeDataBase):
    def execute(
        self,
        *,
        run_id: str,
        client_configurations: typing.List[ClientConfiguration],
        force_regenerate: typing.Optional[bool] = False,
    ):
        dataset_provider, config = self.get_dataset_provider_and_config(run_id=run_id, force_regenerate=force_regenerate)

        self.DATASETS_TO_LOAD = config.automatic_training.features_list

        client_configuration = client_configurations[0]
        client = client_configuration.client

        missing_datasets = []

        for feature_group in self.DATASETS_TO_LOAD:

            df, is_final_df_generated = self.load_data(
                dataset_provider=dataset_provider,
                client=client,
                feature_group=feature_group,
            )

            if is_final_df_generated:
                log.info(f"Dataset {feature_group} already exists. Skipping merging step")
                continue

            if df is None:
                log.info(f"No data found for {feature_group}. Skipping merging step")
                missing_datasets.append(feature_group)
                continue
            
            dataset_provider.set(df=df, dataset_name=feature_group, step=self.CURRENT_STEP)

        self.upload_config(
            dataset_provider=dataset_provider,
            missing_datasets=missing_datasets,
            config=config
        )
