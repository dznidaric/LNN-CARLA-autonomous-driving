
from data_generator import DataGenerator


class CustomDataGenerator(DataGenerator):
    def __init__(self, x_set, y_set, batch_size, shuffle=True, normalize_labels=True):
        super().__init__(x_set, y_set, batch_size, shuffle)
        self.normalize_labels = normalize_labels

    def __getitem__(self, index):
        # Generate one batch of data
        batch_x, batch_y = super().__getitem__(index)

        if self.normalize_labels:
            # Perform label normalization here
            true_max_train = max(abs(min(batch_y)), max(batch_y))
            batch_y *= (1.0 / true_max_train)

        return batch_x, batch_y