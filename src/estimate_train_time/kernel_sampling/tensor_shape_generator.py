import os
import pandas as pd
import math
import numpy as np

class ShapeGenerator:
    def __init__(self, kernel_name: str, precision: str, starts: list, steps: list, ends: list, operators: list, first_n_column: int, raw_data_path: str) -> None:
        self.kernel_name = kernel_name
        self.precision = precision
        self.starts = starts
        self.steps = steps
        self.ends = ends
        # operators is the list of 'multi's or 'add's
        self.operators = operators
        self.first_n_column = first_n_column

        self.raw_data_path = raw_data_path

        self.current_dims = self.get_start_state()
        self.used_dimes = self.get_collected_samples()
        self.total_samples = self.get_total_samples()


    def next_dims(self) -> list:
        if self.current_dims is None and self.used_dimes == 0:
            self.current_dims = self.starts.copy()
            self.used_dimes += 1
            return self.current_dims

        if self.current_dims[0] > self.ends[0]:
            return None

        i = len(self.starts) - 1
        if self.operators[i] == 'add':
            self.current_dims[i] += self.steps[i]
        else:
            self.current_dims[i] *= self.steps[i]

        while i > 0:
            if self.current_dims[i] > self.ends[i]:
                self.current_dims[i] = self.starts[i]
                
                if self.operators[i - 1] == 'add':
                    self.current_dims[i - 1] += self.steps[i - 1]
                else:
                    self.current_dims[i - 1] *= self.steps[i - 1]

                # self.current_dims[i - 1] += self.steps[i - 1]
                i -= 1
            else:
                break

        if self.current_dims[0] <= self.ends[0]:
            self.used_dimes += 1
            return self.current_dims
        else:
            self.current_dims = None
            return None
        
    
    def get_start_state(self):
        # only read the last row from csv file
        if os.path.exists(self.raw_data_path):
            # Get the total number of rows in the file
            total_rows = sum(1 for row in open(self.raw_data_path, 'r'))
            print(f'total_rows: {total_rows}')
            if total_rows > 1:
                # update number of used dimes
                self.used_dimes = total_rows - 1
                # Read only the last row
                last_row = pd.read_csv(self.raw_data_path, skiprows=range(1, total_rows-1)).values.tolist()[0]
                return np.array(last_row[0:self.first_n_column]).astype(int).tolist()
        return None


    def get_total_samples(self):
        total_samples = 1
        for i in range(len(self.starts)):
            if self.operators[i] == 'add':
                total_samples *= (self.ends[i] - self.starts[i]) // self.steps[i] + 1
            else:
                total_samples *= int(math.log(self.ends[i], self.steps[i])) - int(math.log(self.starts[i], self.steps[i])) + 1

        return total_samples


    def get_collected_samples(self):
        if self.current_dims is None:
            return 0

        collected_samples = 0 
        for i in range(len(self.starts)):
            if self.operators[i] == 'add':
                op_times_of_dim = (self.current_dims[i] - self.starts[i]) // self.steps[i]
            else:
                op_times_of_dim = int(math.log(self.current_dims[i], self.steps[i])) - int(math.log(self.starts[i], self.steps[i]))

            # op_times_of_dim = (self.current_dims[i] - self.starts[i]) // self.steps[i]

            j = i + 1
            temp = 1
            while j < len(self.starts):
                
                if self.operators[j] == 'add':
                    temp *= (self.ends[j] - self.starts[j]) // self.steps[j] + 1
                else:
                    temp *= int(math.log(self.ends[j], self.steps[j])) - int(math.log(self.starts[j], self.steps[j])) + 1

                # temp *= (self.ends[j] - self.starts[j]) // self.steps[j] + 1
                j += 1

            collected_samples += op_times_of_dim * temp
            
        collected_samples += 1
        self.used_dimes = collected_samples
        return collected_samples


    def collect_progress(self):
        if self.current_dims is None and self.used_dimes != 0:
            return self.total_samples, self.total_samples
        
        self.used_dimes = self.get_collected_samples()

        return self.used_dimes, self.total_samples


# def test1():
#     kernel_name = 'optimizer'
#     starts = [1, 1]
#     steps = [2, 2]
#     ends = [5, 4]
#     # operators = ['multi', 'add']
#     operators = ['add', 'multi']
#     # operators = ['multi', 'multi']
#     # operators = ['add', 'add']

#     first_n_column = 2
#     precision = 'fp16'

#     shapeGenerator = ShapeGenerator(kernel_name, starts, steps, ends, operators, first_n_column, precision)

#     dims = True
#     print(shapeGenerator.get_total_samples())
#     while dims is not None:
#         dims = shapeGenerator.next_dims()
#         if dims is not None:
#             used_dims, total_samples = shapeGenerator.collect_progress()
#             print(f'{dims}   and   {shapeGenerator.get_collected_samples()}  and  {used_dims}/{total_samples}')


# ### test main() ######


# def test2():
#     kernel_name = 'optimizer'
    
#     # [num_layers, hidden_size]
#     starts = [2, 256]
#     steps = [2, 256]
#     ends = [64, 7168]
#     operators = ['add', 'add']

#     first_n_column = 2
#     precision = 'fp16' 

#     raw_data_path = './sampling_data/NVIDIAA100-SXM4-80GB_optimizer_fp16.csv'

#     shapeGenerator = ShapeGenerator(kernel_name, starts, steps, ends, operators, first_n_column, precision, raw_data_path)

#     print(shapeGenerator.current_dims)


# if __name__ == '__main__':
#     test2()
