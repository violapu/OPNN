import QuantLib as ql

from pricer import BinomialAmericanPricer, BinomialEuropeanPricer, MCAmericanPricer, MCEuropeanPricer, \
    BlackScholesPricer


class PricerFactory:
    def __init__(self, option_type, steps, num_paths, seed=42):
        self.option_type = option_type
        self.steps = steps
        self.num_paths = num_paths
        self.seed = seed

    def create(self, name):
        if name == 'BinomialAmerican':
            return self.create_binomial_american_pricer()
        elif name == 'BinomialEuropean':
            return self.create_binomial_european_pricer()
        elif name == 'MCAmerican':
            return self.create_mc_american_pricer()
        elif name == 'MCEuropean':
            return self.create_mc_european_pricer()
        elif name == 'AnalyticalBS':
            return self.create_analytical_bs_pricer()
        else:
            raise ValueError(f'Invalid pricer name: {name}')

    def create_binomial_american_pricer(self):
        return BinomialAmericanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                      option_type=self.option_type, steps=self.steps)

    def create_binomial_european_pricer(self):
        return BinomialEuropeanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                      option_type=self.option_type, steps=self.steps)

    def create_mc_american_pricer(self):
        return MCAmericanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                option_type=self.option_type, steps=self.steps, num_paths=self.num_paths,
                                seed=self.seed)

    def create_mc_european_pricer(self):
        return MCEuropeanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                option_type=self.option_type, steps=self.steps, num_paths=self.num_paths,
                                seed=self.seed)

    def create_analytical_bs_pricer(self):
        return BlackScholesPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                  option_type=self.option_type)
