from matplotlib import pyplot as plt
import numpy as np

from clasification_mineria import helpers
from clasification_mineria.Clasification import ClassifierFactory, Classifier, \
    TransformerFactory
from clasification_mineria.Readers import ReaderFactory
from clasification_mineria.Tokenizer import TokenizerFactory


class App:
    """
    Attributes:
        _reader (Type[Reader]): Format reader
        _train_path (str): Train folder path
        _test_path (str): Test folder path
    """

    def __init__(self, train_path: str, test_path: str,
                 data_format: str = "standoff"):
        self._reader = ReaderFactory.get_reader(data_format)
        self._train_path = train_path
        self._test_path = test_path

    def start(self, root: str = "out/") -> None:
        """
        Iterates throughout the possible parameters and saves the generated
        analysis files.
        Args:
            root: The folder to be saved the data
        """
        helpers.create_folder(root)
        for tokenizer_name in TokenizerFactory.tokenizers:
            tokenizer = TokenizerFactory.get_tokenizer(tokenizer_name)
            token_path = root + tokenizer_name + "/"
            helpers.create_folder(token_path)
            for model_name in tokenizer.models:
                model_path = token_path + model_name + "/"
                helpers.create_folder(model_path)
                tokenizer_ = tokenizer(model_name)
                rel_types = ['None', 'Lives_In', 'Exhibits']
                for trans_name in TransformerFactory.models:
                    aux = []
                    data = {name: [] for name in ClassifierFactory.names}
                    trans_path = model_path + trans_name + "/"
                    helpers.create_folder(trans_path)
                    for negative_proportion in range(1, 5, 1):
                        prop_path = trans_path + str(negative_proportion) + "/"
                        helpers.create_folder(prop_path)
                        train_dataset = self._reader.load(self._train_path,
                                                          tokenizer_)
                        rel_types = train_dataset.get_relation_types()
                        train_data = train_dataset.get_raw_data(
                            negative_proportion, rel_types)
                        test_data = self._reader.load(self._test_path,
                                                      tokenizer_).get_raw_data(
                            negative_proportion, rel_types)
                        print("train_data", len(train_data["class"]))
                        print("test_data", len(test_data["class"]))
                        for clf_name in ClassifierFactory.names:
                            clf_path = prop_path + clf_name + "/"
                            helpers.create_folder(clf_path)
                            classifier = Classifier(trans_name,
                                                    clf_name)
                            classifier.train(train_data, train_data["class"])
                            print(classifier.len_vocabulary())
                            data_frame = classifier.dev(test_data,
                                                        test_data["class"],
                                                        rel_types,
                                                        clf_path)
                            aux.append((
                                f"{clf_name}_1:{negative_proportion}",
                                data_frame))
                    fig1 = plt.figure()
                    ax1 = fig1.add_subplot(111)
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)
                    ax1.set_title("Varianza en las relaciones positivas"
                                  " en proporcion a las relaciones negativas",
                                  fontsize=10)
                    ax1.set_xlabel('Metricas')
                    ax1.set_ylabel('Valor de la metrica (0-1)')
                    ax2.set_title("Varianza en las relaciones positivas"
                                  " en proporcion a las relaciones negativas",
                                  fontsize=10)
                    ax2.set_xlabel('Proporcion')
                    ax2.set_ylabel('Valor del f-score (0-1)')
                    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(aux))))
                    rel_types.remove("None")
                    for name, data_frame in aux:
                        rel_df = data_frame[rel_types]
                        weights = rel_df.loc["support"].values
                        rel_df = rel_df.drop("support")
                        rel_df *= weights
                        ser = rel_df.sum(axis=1) / sum(weights)
                        ax1.plot(ser.index, ser, color=next(color),
                                 label=name)
                        data[name.split("_")[0]].append(ser["f1-score"])
                    x = list(range(1, 5, 1))
                    ind = np.arange(float(len(x)))
                    width = 0.35
                    for key in data:
                        ax2.bar(ind, data[key], width, label=key)
                        ind += width
                    plt.xticks(np.arange(len(x)) + width / 2,
                               tuple(map(lambda z: f"1:{str(z)}", x)))
                    ax1.legend()
                    ax2.legend(title="Models")
                    fig1.savefig(trans_path + "proportions_variance_all.png")
                    fig2.savefig(trans_path + "proportion_variance_fscore.png")
                    plt.close()
