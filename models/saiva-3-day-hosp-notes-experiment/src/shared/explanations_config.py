exp_dictionary = {
    'Diet_Order': {
            7: ["nepro", "nepro (oral)","npo", "not applicable-npo only", "puree", "pureed", "puree with ground meats",
                "prostat sf awc", "pro-stat 101", "pro-stat 64", "pro-stat awc", "pro-stat sugar free", "pro-stat profile",
                "prostat sugar free", "prostat liquid", "prostat, 64", "prostat awc", "calorically dense oral supplement",
                "mechanical soft", "moist mechanical soft", "mechanical soft with ground meats"],
            14: ["npo", "not applicable-npo only", "puree", "pureed", "puree with ground meats","calorically dense oral supplement",
                 "mechanical soft", "moist mechanical soft", "mechanical soft with ground meats"],
            30: ["npo", "not applicable-npo only", "puree", "pureed", "puree with ground meats", "mechanical soft", 
                 "moist mechanical soft", "mechanical soft with ground meats"]
    },
    'Diagnostic_Order': [7, 14, 30],
    'Alert_Count_Indicator': {
                7: []
        },
    'Alert_Indicator': {
                7: ["more than 2 episodes of loose bm's", "resident has not had a bm within 48 hours",
                    "resident refused bathing", "resident ate less than 25 percent of meal", "change in therapy level - decrease"],
                14: ["more than 2 episodes of loose bm's", "resident refused bathing"]
    },
    'Patient_Meds': {
                7: [ "loop diuretics", "bulk chemicals - a's", "sympathomimetics", "5-ht3 receptor antagonists",
                     "phosphate binder agents", "b-complex w/ folic acid", "0500000000", "sodium", "insulin",
                     "misc. topical", "antineoplastic - hormonal and related agents", "misc. respiratory inhalants",
                     "cephalosporins - 1st generation", "dibenzapines", "5300000000", "3800000000", "iron",
                     "hmg coa reductase inhibitors", "dibenzapines", "platelet aggregation inhibitors", "cardiac glycosides",
                     "hepatitis agents", "hematopoietic growth factors", "b-complex w/ c", "coumarin anticoagulants",
                     "0700000000", "laxative combinations", "stimulant laxatives", "alpha-beta blockers",
                     "antidiarrheal/probiotic agents - misc.", "magnesium"],
                14: ["loop diuretics", "bulk chemicals - a's", "sympathomimetics", "5-ht3 receptor antagonists",
                     "phosphate binder agents", "b-complex w/ folic acid", "0500000000", "sodium", "insulin",
                     "misc. topical", "antineoplastic - hormonal and related agents", "misc. respiratory inhalants",
                     "cephalosporins - 1st generation", "dibenzapines", "5300000000", "3800000000", "iron",
                     "hmg coa reductase inhibitors", "dibenzapines", "platelet aggregation inhibitors", "cardiac glycosides",
                     "hepatitis agents", "hematopoietic growth factors", "b-complex w/ c", "coumarin anticoagulants",
                     "0700000000","laxative combinations", "stimulant laxatives", "alpha-beta blockers",
                     "antidiarrheal/probiotic agents - misc.", "magnesium"],
                30: ["loop diuretics", "bulk chemicals - a's", "sympathomimetics", "5-ht3 receptor antagonists",
                     "phosphate binder agents", "b-complex w/ folic acid", "0500000000", "sodium", "insulin",
                     "misc. topical", "antineoplastic - hormonal and related agents", "misc. respiratory inhalants",
                     "cephalosporins - 1st generation", "dibenzapines", "5300000000", "3800000000", "iron",
                     "hmg coa reductase inhibitors", "dibenzapines", "platelet aggregation inhibitors", "cardiac glycosides",
                     "hepatitis agents", "hematopoietic growth factors", "b-complex w/ c", "coumarin anticoagulants", "0700000000"],

    },
    'Patient_Vitals': {
                'Maximum': {'pulse':(109,), 'diastolicvalue':(92,200), 'temperature':(99.9,108.6), 'blood sugar':(245,525), 'respiration':(27,70)},
                'Minimum': {'diastolicvalue':(25,70), 'o2 sats':(70,90), 'temperature':(85,97.5), 'blood sugar':(15,70)},
                'Others':{'bmi':(34,)}
    },
    'Patient_Diagnosis': {
                'all':["diseases of the blood and blood-forming organs - anemia",
                       "diseases of the circulatory system",
                       "endocrine; nutritional; and metabolic diseases and immunity disorders",
                       "neoplasms",
                       "diseases of the blood and blood-forming organs - coagulation and hemorrhagic disorders [62.]",
                       "endocrine; nutritional; and metabolic diseases and immunity disorders - other nutritional; endocrine; and metabolic disorders [58.]",
                       "mental illness",
                       "certain conditions originating in the perinatal period - intrauterine hypoxia and birth asphyxia [220.]",
                       "endocrine; nutritional; and metabolic diseases and immunity disorders - diabetes mellitus with complications [50.]",
                       "congenital anomalies",
                       "diseases of the skin and subcutaneous tissue",
                       "diseases of the musculoskeletal system and connective tissue - pathological fracture [207.]",
                       "diseases of the musculoskeletal system and connective tissue - systemic lupus erythematosus and connective tissue disorders [210.]",
                       "diseases of the musculoskeletal system and connective tissue - acquired deformities"],
                7: ["diseases of the musculoskeletal system and connective tissue - non-traumatic joint disorders"],
                14: ["diseases of the musculoskeletal system and connective tissue - non-traumatic joint disorders"],
                30: ["diseases of the musculoskeletal system and connective tissue - other connective tissue disease [211.]",
                     "diseases of the digestive system",
                     "injury and poisoning",
                     "diseases of the musculoskeletal system and connective tissue - infective arthritis and osteomyelitis (except that caused by tb or std) [201.]",
                     "residual codes; unclassified; all e codes [259. and 260.] -",
                     "endocrine; nutritional; and metabolic diseases and immunity disorders - nutritional deficiencies [52.]",
                     "infectious and parasitic diseases",
                     "diseases of the genitourinary system",
                     "diseases of the nervous system and sense organs",
                     "diseases of the respiratory system"]
    },
    'Patient_Labs': {
        'all_labs': ['abnormal - (non-numeric)', 'above high normal', 'above highest instrument scale', 'above upper panic limits',
                     'below low normal', 'below lower panic limits', 'below lowest instrument scale', 'better', 'intermediate',
                     'moderately susceptible', 'normal', 'resistant', 'significant change down', 'significant change up',
                     'susceptible', 'very abnormal - (non-numeric)', 'very susceptible', 'worse'],
        'labs_with_values':['above high normal', 'below low normal', 'better', 'intermediate', 'moderately susceptible',
                            'normal', 'significant change down', 'significant change up', 'susceptible','very susceptible',
                            'worse']
    }
}

