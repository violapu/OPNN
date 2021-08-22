import pandas as pd
from typing import List


class History:
    def __init__(self, iteration: int, interior_loss: float, boundary_loss: float,
                 interior_validation_loss: float, boundary_validation_loss: float,
                 interior_point_count: int, boundary_point_count: int, time_elapsed: float):
        self.iteration = iteration
        self.interior_loss = interior_loss
        self.interior_validation_loss = interior_validation_loss
        self.boundary_loss = boundary_loss
        self.boundary_validation_loss = boundary_validation_loss
        self.interior_point_count = interior_point_count
        self.boundary_point_count = boundary_point_count
        self.time_elapsed = time_elapsed

    def __repr__(self):
        return f'Iteration: {self.iteration}, Interior loss: {self.interior_loss}, ' \
               f'Interior validation loss: {self.interior_validation_loss}, ' \
               f'Boundary loss: {self.boundary_loss}, ' \
               f'Boundary validation loss: {self.boundary_validation_loss}, ' \
               f'point count: {self.interior_point_count}, ' \
               f'Boundary point count: {self.boundary_point_count}, Time elapsed: {self.time_elapsed}'

    @staticmethod
    def save_history(history: List['History'], l2_error: float, max_error: float, path: str):
        iters = []
        int_losses = []
        int_validation_losses = []
        boundary_losses = []
        boundary_validation_losses = []
        interior_point_counts = []
        boundary_point_counts = []
        times_elapsed = []

        for h in history:
            iters.append(h.iteration)
            int_losses.append(h.interior_loss)
            int_validation_losses.append(h.interior_validation_loss)
            boundary_losses.append(h.boundary_loss)
            boundary_validation_losses.append(h.boundary_validation_loss)
            interior_point_counts.append(h.interior_point_count)
            boundary_point_counts.append(h.boundary_point_count)
            times_elapsed.append(h.time_elapsed)

        df = pd.DataFrame()
        df['iterations'] = iters
        df['int_losses'] = int_losses
        df['int_validation_losses'] = int_validation_losses
        df['boundary_losses'] = boundary_losses
        df['boundary_validation_losses'] = boundary_validation_losses
        df['interior_point_counts'] = interior_point_counts
        df['boundary_point_counts'] = boundary_point_counts
        df['times_elapsed'] = times_elapsed
        df['l2_error'] = l2_error
        df['max_error'] = max_error

        df.to_csv(path, index=False)
