import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator


class Walklets(Estimator):


    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 32,
        workers: int = 4,
        window_size: int = 4,
        epochs: int = 1,
        use_hierarchical_softmax: bool = True,
        number_of_negative_samples: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.use_hierarchical_softmax = use_hierarchical_softmax
        self.number_of_negative_samples = number_of_negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def _select_walklets(self, walks, power):
        walklets = []
        for walk in walks:
            for step in range(power + 1):
                neighbors = [n for i, n in enumerate(walk[step:]) if i % power == 0]
                walklets.append(neighbors)
        return walklets

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Walklets model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)
        num_of_nodes = graph.number_of_nodes()

        self._embedding = []
        for power in range(1, self.window_size + 1):
            walklets = self._select_walklets(walker.walks, power)
            model = Word2Vec(
                walklets,
                hs=1 if self.use_hierarchical_softmax else 0,
                negative=self.number_of_negative_samples,
                alpha=self.learning_rate,
                epochs=self.epochs,
                vector_size=self.dimensions,
                window=1,
                min_count=self.min_count,
                workers=self.workers,
                seed=self.seed,
            )

            embedding = np.array([model.wv[str(n)] for n in range(num_of_nodes)])
            self._embedding.append(embedding)

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.concatenate(self._embedding, axis=1)
