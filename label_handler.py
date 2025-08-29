def parse_labels(y_dict, scp_mapping):
    """
    Convert raw scp_codes dict into list of labels
    scp_mapping: maps diagnostic codes -> category
    """
    labels = []
    for key in y_dict.keys():
        if key in scp_mapping:
            labels.append(scp_mapping[key])
    return labels
