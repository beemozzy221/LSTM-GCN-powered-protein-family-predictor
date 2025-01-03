import numpy as np
from sklearn.model_selection import train_test_split

class TrainTestVaiCreate:

    def __init__(self,x_array: np.ndarray,y_array: np.ndarray):
        self.x_array = x_array
        self.y_array = y_array
    
    def split(self, test_size):
        """Splits data into train/val/test sets and normalizes the data.

        Args:
            test_size: Dataset split ratio, float

        Returns:
            `train_array`, `val_array`, `test_array`
        """

        xdata = self.x_array
        ydata = self.y_array
        indices = np.arange(len(xdata))

        x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
            xdata, ydata, indices, test_size=test_size, random_state=42
        )

        xml_data = (x_train, x_test, train_indices, test_indices)
        yml_data = (y_train, y_test, train_indices, test_indices)

        return xml_data, yml_data






    

