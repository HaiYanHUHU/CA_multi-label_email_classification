from stuff.data import get_input_data, pre_process
from stuff import Dataset
from stuff.models import RandomF, Logistic, DNN
from utils import decode_classification_report, print_classification_report
import warnings


def main():
    warnings.filterwarnings('ignore')

    df = get_input_data(['data/AppGallery.csv', 'data/Purchasing.csv'])
    df = pre_process(df)

    for y1, subdf in df.groupby('y1'):
        ds = Dataset(subdf, ["summary", "content"])
        enc = ds.tuplefy(('y2', 'y3', 'y4'))
        ds.vectorise("summary", 1000)
        ds.vectorise("content", 5000)
        ds.split()

        classifiers = [RandomF(ds, n_estimators=10000), Logistic(ds), DNN(ds, (512,))]

        for clf in classifiers:
            clf.train()

            print('Evaluation for ' + y1 + ', ' + clf.name)

            dc = decode_classification_report(clf.evaluate(dict=True), enc)
            print_classification_report(dc)

if __name__ == '__main__':
    main()