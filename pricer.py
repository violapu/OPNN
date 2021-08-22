import numpy as np
import QuantLib as ql
import quantsbin.derivativepricing as qbdp

from datetime import datetime, timedelta
from enums import OptionType
from generator import STRIKE_COL, UNDERLYING_COL, RF_RATE_COL, DAYS_TO_MATURITY_COL, DIV_COL, SIGMA_COL
from tqdm import tqdm

"""
 Available pricers are BinomialAmericanPricer, BinomialEuropeanPricer, MCAmericanPricer, 
 MCEuropeanPricer, BlackScholesPricer, DiscreteDividendsPricer, AnalyticalHestonPricer,
 MCEuropeanHestonEngine.
"""


def bsm(underlying, rf_rate, div, sigma, day_count, calendar, calculation_date):
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, rf_rate, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, div, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, sigma, day_count))

    return ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)


class Pricer:
    def price(self, df, use_tqdm=False):
        prices = np.zeros(df.shape[0])

        iterator = df.iterrows()
        if use_tqdm:
            iterator = tqdm(iterator, total=df.shape[0])

        for i, row in iterator:
            strike = row[STRIKE_COL]
            sigma = row[SIGMA_COL]
            div = row[DIV_COL]
            days_to_maturity = row[DAYS_TO_MATURITY_COL]
            underlying = row[UNDERLYING_COL]
            rf_rate = row[RF_RATE_COL]

            prices[i] = self.price_one(strike, sigma, div, days_to_maturity, underlying, rf_rate)

        return prices

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        pass


class BinomialAmericanPricer(Pricer):
    def __init__(self, use_tqdm, calculation_date, option_type, steps):
        Pricer.__init__(use_tqdm)
        self.calculation_date = calculation_date
        self.option_type = OptionType.ql_type(option_type)
        self.steps = steps

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        maturity_date = ql.Date(self.calculation_date.serialNumber() + int(maturity))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date

        process = bsm(underlying, rf_rate, div, sigma, day_count, calendar, self.calculation_date)

        payoff = ql.PlainVanillaPayoff(self.option_type, strike)
        exercise = ql.AmericanExercise(self.calculation_date, maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        binomial_engine = ql.BinomialVanillaEngine(process, 'crr', self.steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()


class BinomialEuropeanPricer(Pricer):
    def __init__(self, use_tqdm, calculation_date, option_type, steps):
        Pricer.__init__(use_tqdm)
        self.calculation_date = calculation_date
        self.option_type = OptionType.ql_type(option_type)
        self.steps = steps

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        maturity_date = ql.Date(self.calculation_date.serialNumber() + int(maturity))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date

        process = bsm(underlying, rf_rate, div, sigma, day_count, calendar, self.calculation_date)

        payoff = ql.PlainVanillaPayoff(self.option_type, strike)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        binomial_engine = ql.BinomialVanillaEngine(process, 'crr', self.steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()


class MCAmericanPricer(Pricer):
    def __init__(self, use_tqdm, calculation_date, option_type, steps, num_paths, seed):
        Pricer.__init__(use_tqdm)
        self.calculation_date = calculation_date
        self.option_type = OptionType.ql_type(option_type)
        self.steps = steps
        self.num_paths = num_paths
        self.seed = seed

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        maturity_date = ql.Date(self.calculation_date.serialNumber() + int(maturity))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date

        process = bsm(underlying, rf_rate, div, sigma, day_count, calendar, self.calculation_date)

        payoff = ql.PlainVanillaPayoff(self.option_type, strike)
        exercise = ql.AmericanExercise(self.calculation_date, maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        mc_engine = ql.MCAmericanEngine(process, 'pseudorandom', self.steps,
                                        requiredSamples=self.num_paths, seed=self.seed)
        option.setPricingEngine(mc_engine)
        return option.NPV()


class MCEuropeanPricer(Pricer):
    def __init__(self, use_tqdm, calculation_date, option_type, steps, num_paths, seed):
        Pricer.__init__(use_tqdm)
        self.calculation_date = calculation_date
        self.option_type = OptionType.ql_type(option_type)
        self.steps = steps
        self.num_paths = num_paths
        self.seed = seed

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        maturity_date = ql.Date(self.calculation_date.serialNumber() + int(maturity))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date

        process = bsm(underlying, rf_rate, div, sigma, day_count, calendar, self.calculation_date)

        payoff = ql.PlainVanillaPayoff(self.option_type, strike)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        mc_engine = ql.MCEuropeanEngine(process, 'pseudorandom', self.steps,
                                        requiredSamples=self.num_paths, seed=self.seed)
        option.setPricingEngine(mc_engine)
        return option.NPV()


class BlackScholesPricer(Pricer):
    def __init__(self, use_tqdm, calculation_date, option_type):
        Pricer.__init__(use_tqdm)
        self.calculation_date = calculation_date
        self.option_type = OptionType.ql_type(option_type)

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        maturity_date = ql.Date(self.calculation_date.serialNumber() + int(maturity))

        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = self.calculation_date

        process = bsm(underlying, rf_rate, div, sigma, day_count, calendar, self.calculation_date)

        payoff = ql.PlainVanillaPayoff(self.option_type, strike)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        bs_engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(bs_engine)
        return option.NPV()


class DiscreteDividendsPricer(Pricer):
    def __init__(self, use_tqdm, pricing_date, option_type, expiry_type, method):
        Pricer.__init__(use_tqdm)
        self.pricing_date = pricing_date
        self.option_type = OptionType.ql_type(option_type)
        self.expiry_type = expiry_type
        self.method = method

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        pricing_date = datetime.strptime(self.pricing_date, '%Y%m%d')
        expiry_date = (pricing_date + timedelta(days=maturity)).strftime('%Y%m%d')
        div_list = [((pricing_date + timedelta(days=int(days))).strftime('%Y%m%d'), amount) for
                    days, amount
                    in div.transpose()]
        option = qbdp.EqOption(option_type=self.option_type, strike=strike, expiry_date=expiry_date,
                               expiry_type=self.expiry_type)
        parameters = {'spot0': underlying,
                      'pricing_date': self.pricing_date,
                      'volatility': sigma,
                      'rf_rate': rf_rate,
                      'div_list': div_list}
        option_model = option.engine(model=self.method, **parameters)
        return option_model.valuation()
