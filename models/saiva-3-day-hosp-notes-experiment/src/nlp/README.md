# Saiva NLP

- `python build_ner_model.py --version model_feature`

   used when ever we want to retrain the model
with new keywords by updating the ./keywords/model_feature_keywords.csv file.
Once the model is bundled, its stored in S3.

- `build_topic_model.py` is consumed by Jupyter Notebook train-topic-model.ipynb.
This script is used when ever we want to retrain the Topic model
and save it directly  in S3.

- `load.py` is used to load the model stored in S3

- `utils.py` will all common functions which is used in multiple places