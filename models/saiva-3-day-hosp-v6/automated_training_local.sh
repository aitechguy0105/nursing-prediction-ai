#!/usr/bin/env bash
set -e

readJsonConfig() {
  echo $json | jq -r $1
}

format_array() {
  if [ "$1" == "null" ]; then
    echo "null"
    return
  fi

  echo $@

  str=$(echo $@ | sed -e 's/\[ //g' -e 's/\ ]//g' -e 's/\,//g')

  arr=( $str )

  echo "arr = ${arr[@]}"

  res="\"["

  for v in "${!arr[@]}"
  do
    if [ $v -eq 0 ]; then
      res+="${arr[$v]}"
    else
      res+=",${arr[$v]}"
    fi
  done

  res="$res]\""

  echo $res
}

format_array_of_objects() {
  if [ "$1" == "null" ]; then
    echo "null"
    return
  fi

  # remove only spaces
  str=$(echo $@ | sed -e 's/ //g')

  echo "'$str'"
}

run_step() {
  echo "Running step $step"

  IFS='.' read -r -a split_step <<< "$step"

  if [ ${split_step[0]} == "datacards" ]; then
    make $make_target_automatic_training command="python /src/training_pipeline/training_steps/datacard_${split_step[1]}.py $configured_params"

    FILE=data/datacard_${split_step[1]}_params.txt
    if [ -f "$FILE" ]; then
      datacard_params=$(cat $FILE)

      sudo rm -f $FILE

      # split datacard_params by new line
      IFS=$'\n' datacard_params=($datacard_params)

      # for loop that iterates over each element in datacard_params
      for i in "${!datacard_params[@]}"
      do
        # store aws command output in variable
        docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) 570873190097.dkr.ecr.us-east-1.amazonaws.com
        docker pull 570873190097.dkr.ecr.us-east-1.amazonaws.com/saiva-datacards:$datacards_image_env
        docker_command="docker run --rm -t -e SAIVA_ENV=$env 570873190097.dkr.ecr.us-east-1.amazonaws.com/saiva-datacards:$datacards_image_env python run_datacard.py ${split_step[1]} run ${datacard_params[$i]}"
        clean_command=$(echo "$docker_command" | sed "s/'//g")
        eval "$clean_command"
      done
    fi

  elif [ ${#split_step[@]} -eq 2 ]; then
    make $make_target_automatic_training command="python /src/training_pipeline/training_steps/${split_step[0]}.py $configured_params --feature ${split_step[1]}"
  else
    make $make_target_automatic_training command="python /src/training_pipeline/training_steps/$step.py $configured_params"
  fi
}

for ARGUMENT in "$@"
    do
      KEY=$(echo $ARGUMENT | cut -f1 -d=)

      KEY_LENGTH=${#KEY}
      VALUE="${ARGUMENT:$KEY_LENGTH+1}"

      export "$KEY"="$VALUE"
    done
# use here your expected variables
echo "env = $env"
echo "datacards_image_env = $datacards_image_env"
echo "command = $command"
echo "step = $step"
echo "start_step = $start_step"
echo "use_amd64 = $use_amd64"

if [ "$use_amd64" == "true" ]; then
  make_target_automatic_training="run_automatic_training_amd64_local_script"
  make_target_datacards="run_amd64"
else
  make_target_automatic_training="run_automatic_training_local_script"
  make_target_datacards="run"
fi

declare -a training_steps=(
  "setup" 
  "configure_facilities" 
  "calculate_date_range" 
  "fetch_data"
  "preprocess_data"
  "datacards.facility_discovery"
  "datacards.data_availability"
  "merge_data"
  "check_amount_of_data"
  "dates_calculation"
  "feature_engineering.patient_census"
  "feature_engineering.alerts"
  "feature_engineering.demographics"
  "feature_engineering.lab_results"
  "feature_engineering.meds"
  "feature_engineering.orders"
  "feature_engineering.patient_admissions"
  "feature_engineering.patient_assessments"
  "feature_engineering.patient_diagnosis"
  "feature_engineering.patient_immunizations"
  "feature_engineering.patient_progress_notes"
  "feature_engineering.patient_rehosps"
  "feature_engineering.patient_risks"
  "feature_engineering.vitals"
  "feature_engineering.patient_adt"
  "post_feature_engineering_merge"
  "feature_selection"
  "datasets_generation"
  "datacards.xaedy"
  "data_distribution"
  "train_model"
  "datacards.prediction_probability"
  "datacards.shap_values"
  "datacards.decisions"
  "datacards.trained_model"
  "upload_model_metadata")

json=$(cat training_params.json)

declare -A params=(
  ["run_id"]=$(readJsonConfig ".run_id")
  ["force_regenerate"]=$(readJsonConfig ".force_regenerate")
  ["experiment_dates"]=$(format_array_of_objects $(readJsonConfig ".experiment_dates"))
  ["optuna_time_budget"]=$(readJsonConfig ".optuna_time_budget")
  ["model_type"]=$(readJsonConfig ".model_type")
  ["client_configurations"]=$(format_array_of_objects $(readJsonConfig ".client_configurations"))
  ["hyper_parameter_tuning"]=$(readJsonConfig ".hyper_parameter_tuning")
  ["invalid_action_types"]=$(format_array $(readJsonConfig ".invalid_action_types"))
  ["required_features"]=$(format_array $(readJsonConfig ".required_features"))
  ["required_features_preprocess"]=$(format_array $(readJsonConfig ".required_features_preprocess"))
  ["strategy"]=$(readJsonConfig ".strategy")
)

if [ "${params["force_regenerate"]}" == "false" ]; then
  params["force_regenerate"]="null"
fi

if [ "${params["hyper_parameter_tuning"]}" == "false" ]; then
  params["hyper_parameter_tuning"]="null"
fi

params["disable_sentry"]="true"

configured_params=""

for param in "${!params[@]}"
do
  value="${params[$param]}"
  if [ "$value" != "null" ]; then
    configured_params+=" --$param $value"
  fi
done

echo "configured_params = $configured_params"

start_index=0

if [ "$start_step" != "" ]
then
  for i in "${!training_steps[@]}"; do
    [[ "${training_steps[$i]}" = "${start_step}" ]] && break
    ((start_index+=1))
  done
fi

if [ $start_index -eq ${#training_steps[@]} ]; then
  echo "Start step $start_step not found"
  exit 1
fi

if [ "$command" == "" ] && [ "$step" == "" ]
  then
    for (( step_idx=$start_index; step_idx<${#training_steps[@]}; step_idx++ ));
    do
      step=${training_steps[$step_idx]}
      run_step
    done
  else
    if [ "$command" != "" ]; then
      # Call the command and store the returned value in variable
      make $make_target_automatic_training command="$command"
    elif [ "$step" != "" ]; then
        run_step
    fi
fi