
def decode_classification_report(report_dict, encoder):
    """
    decode the labels of a classification report dictionary using an sklearn-like encoder
    """
    decoded_report = {}
    for key, value in report_dict.items():
        try:
            decoded_key = str(encoder.inverse_transform([int(key)])[0])
        except (ValueError, KeyError):
            decoded_key = key  # Keep keys like 'accuracy', 'macro avg', etc.
        decoded_report[decoded_key] = value
    return decoded_report


def print_classification_report(report_dict):
    """
    format an sklear-like classification report
    """
    headers = ["precision", "recall", "f1-score", "support"]
    max_label_length = max(len(str(label)) for label in report_dict.keys())

    print("\n" + " " * (max_label_length + 2) + "precision    recall  f1-score   support")
    print("".ljust(max_label_length + 40, "-"))

    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(
            f"{str(label):<{max_label_length}}  {metrics['precision']:9.2f} {metrics['recall']:9.2f} {metrics['f1-score']:9.2f} {metrics['support']:9.0f}")

    print("".ljust(max_label_length + 40, "-"))

    if "accuracy" in report_dict:
        print(
            f"accuracy{''.ljust(max_label_length - 8)}  {report_dict['accuracy']:9.2f} {sum(v['support'] for k, v in report_dict.items() if isinstance(v, dict)):9.0f}")
        print("".ljust(max_label_length + 40, "-"))

    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report_dict:
            avg_metrics = report_dict[avg_type]
            print(
                f"{avg_type:<{max_label_length}}  {avg_metrics['precision']:9.2f} {avg_metrics['recall']:9.2f} {avg_metrics['f1-score']:9.2f} {avg_metrics['support']:9.0f}")
