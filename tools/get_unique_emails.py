import fire

AVANTE_REPORT_CONFIG = {
    3: {
        'emails': ["wmilam@avantegroup.com", "JHand@avantecenters.com", "SJoubert@avantecenters.com",
                   "Rruohonen@avantecenters.com", "KRatanasurakarn@avantegroup.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    4: {
        'emails': ["wmilam@avantegroup.com", "psurtain@avantecenters.com", "Asfoster@avantecenters.com",
                   "Lerobinson@avantecenters.com", "Ejeanpierre@avantecenters.com"],
        'model_id': 'e9943c54f0e14c04ba59d1bf53120bf9'
    },
    5: {
        'emails': ["kervin@avantecenters.com", "sbruno@avantegroup.com", "wmilam@avantecenters.com",
                   "tbenjamin@avantecenters.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    6: {
        'emails': ["wmilam@avantegroup.com", "awoodman@avantecenters.com", "bicrawford@avantecenters.com",
                   "BNunn@avantecenters.com", "sbruno@avantegroup.com"],
        'model_id': '9ee8d86ae829409eaecffddf6634d605'
    },
    7: {
        'emails': ["clawrenceRice@avantegroup.com",
                   "wmilam@avantegroup.com"],
        'model_id': 'e9943c54f0e14c04ba59d1bf53120bf9'
    },
    8: {
        'emails': ["wmilam@avantegroup.com", "GStrunk-Gamel@avantecenters.com", "mmcdowell@avantecenters.com",
                   "KRatanasurakarn@avantegroup.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    9: {
        'emails': ["rbachman@avantecenters.com", "kratanasurakarn@avantegroup.com", "wmilam@avantecenters.com",
                   "maugustin@avantecenters.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    10: {
        'emails': ["wmilam@avantegroup.com", "sbruno@avantegroup.com", "kApodaca@avantecenters.com",
                   "MVelezMaldonado@avantecenters.com", "nsierra@avantecenters.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    13: {
        'emails': ["AMuniz@avantecenters.com", "wmilam@avantegroup.com", "clawrenceRice@avantegroup.com",
                   "ljames@avantecenters.com", "cescander@avantecenters.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    21: {
        'emails': ["wmilam@avantegroup.com", "mgeorge@avantecenters.com", "Afortner@avantecenters.com",
                   "sbruno@avantegroup.com", "nkypriotes@avantecenters.com"],
        'model_id': '4e78363600a14e65866d8c1ef7ab28fe'
    },
    1: {
        'emails': ["wmilam@avantegroup.com", "utheoc@avantecenters.com", "SHenriques@avantecenters.com",
                   "cparramore@avantecenters.com"],
        'model_id': '9ee8d86ae829409eaecffddf6634d605'
    },
}

def get_unique_emails():
    unique_emails = set()
    for values in AVANTE_REPORT_CONFIG.values():
        for email in values['emails']:
            unique_emails.add(email)
    joined_emails = ', '.join(unique_emails)
    print(joined_emails)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(get_unique_emails)

# examples
# python get_unique_emails.py