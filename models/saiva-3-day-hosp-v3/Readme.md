# saiva-3-day-hosp-v3 model

## Content
1. Similar to v1 model feature group
2. Admission feature group is newly added 
3. Restructured the code for feature engineering. 
   Each feature group has its own file under `shared/` folder. 
4. All feature groups are merged in `data_manager.py` file.
5. Training notebooks have additional 05-feature-selection.ipynb notebook. 
   Here we drop features having >= 98% zeros.
   