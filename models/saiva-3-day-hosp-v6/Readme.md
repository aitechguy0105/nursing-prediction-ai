# saiva-3-day-hosp-v6 model

## Content
1. Ranked residents during training ie. exclude hospice, certain census codes
2. Consider only RTHs to minimise and discard Excluded RTHs
3. Bug fix : predict for next 3 days
4. Get all patient admissions & transfers from the begining 
5. In Admission featurization change the bins to 7 to 1 rather than 6 to 0.
6. In Transfers featurization change days_since_last_hosp feature to binning technique
7. Remove measuring of AUC at a facility level, rather we have one single AUC score for the entire test set. 
8. To make the model smarter add facilityid as categorical field
9. Training, validation & Test data needs to be similar to real time prediction data. 
   ie. during real time prediction for any given day we have data upto yesterday night
10. Verify the RTH distribution in 06a before model training

## Installation notes

### MacOS (arm64)

1. Install [Homebrew](https://brew.sh/)
2. Install `libomp` with `brew install libomp`
   - this is required for `lightgbm` to work
3. Install pipenv dependencies* with `CFLAGS="-mavx -DWARN(a)=(a)" pipenv install`

* - Currently, library `nmslib` (version 2.1.1) is not working on MacOS (arm64) even though arm64 binaries are available and should be used. This is a known issue and is being tracked.
As a workound, we have to toggle some of the flags for compiler to make it work. This is done by setting `CFLAGS` environment variable.