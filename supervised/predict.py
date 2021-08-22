import ast
import pandas as pd
import time

from generator import build_constant_generator


def predict(problem, prediction_params, unique_name):
    domain = ast.literal_eval(prediction_params['domain'])
    gen = build_constant_generator(domain)
    prices_df = gen.generate(samples=int(prediction_params['samples']))

    start_time = time.time()
    supervised_prices = problem.predict(prices_df)
    end_time = time.time()
    print('time elapsed: ', end_time - start_time)

    pd.DataFrame(supervised_prices).to_csv(f'comparison_data/supervised_{unique_name}.csv', index=False)
