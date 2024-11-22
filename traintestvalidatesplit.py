import numpy as np
from sklearn.model_selection import train_test_split

class ttvcreate:

    
    
    def __init__(self,x_array: np.ndarray,y_array: np.ndarray):
        self.xarray = x_array
        self.yarray = y_array
    
    def preprocess(self, test_size: float):
        """Splits data into train/val/test sets and normalizes the data.

        Args:
            data_array: ndarray of shape `(num_time_steps, num_routes)`
            train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
                to include in the train split.
            val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
                to include in the validation split.

        Returns:
            `train_array`, `val_array`, `test_array`
        """

        xdata = self.xarray

        ydata = self.yarray

        indices = np.arange(len(xdata))

        x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
            xdata, ydata, indices, test_size=0.3, random_state=42
        )

        xmldata = (x_train, x_test, train_indices, test_indices)

        ymldata = (y_train, y_test, train_indices, test_indices)

        return xmldata, ymldata






    

