import anago
from anago.utils import load_data_and_labels


if __name__ == "__main__":
    model = anago.Sequence()
    x_train, y_train = load_data_and_labels('train.txt')
    model.fit(x_train[100:], y_train[100:], epochs=1)

    # test
    model.score(x_train[:100], y_train[:100])


