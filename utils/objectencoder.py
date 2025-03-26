import typing as t

class ObjectEncoder:
    classes_: t.Mapping[int, object] = None
    classes_int_: t.Mapping[object, int] = None

    def __init__(self):
        pass

    def fit(self, X: t.Iterator[object]):
        tmp = []
        for obj in X:
            if obj not in tmp:
                tmp.append(obj)

        self.classes_ = tuple(tmp)
        self.classes_int_ = {obj: i for i, obj in enumerate(self.classes_)}

        return self

    def transform(self, X: t.Iterator[object]) -> list:
        return [self.classes_int_[x] for x in X.__iter__()]

    def fit_transform(self, X: t.Iterator[object]) -> list:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: t.Iterator[int]) -> list:
        return [self.classes_[x] for x in X]