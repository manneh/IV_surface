import numpy as np
from math import sqrt, exp, log, pi
from scipy.stats import norm

class Option:
    def __init__(self, asset_price, strike_price, call_price, 
    rf_rate, T, put_price):
        self.strike_price = strike_price
        self.call_price = call_price
        self.asset_price = asset_price
        self.put_price = put_price
        self.rf_rate = rf_rate
        self.T = T

    def spot_from_put_call(self):
        PV_strike = self.strike_price * (1.0 + self.rf_rate)**(-self.T)
        return self.call_price + PV_strike - self.put_price

    def forward_price(self, c=0, t_income=0):
        """This Function calculates the forward price of an underlying asset]
         given its cash floww and spot price.

        Keyword Arguments:
            c {float or ndarray} -- the amount of cash flow (default: {0})
            t_income {int or ndarray} -- the timestemps when the CF
             will be delivered (default: {0})

        Returns:
            [float] -- [the forward price]
        """

        cf_incomes = c * np.exp(self.rf_rate * (self.T - t_income))
        zero_coupon_forward =\
            self.spot_from_put_call() * np.exp(self.rf_rate * self.T)

        return zero_coupon_forward - cf_incomes

    def logforward_moneyness(self):
        return np.log(self.strike_price / self.forward_price())

    def smile_tanh(self, eps=1e-6):
        m = self.logforward_moneyness()
        return np.sqrt(m * np.tanh(0.5+m) + np.tanh(eps - 0.5 * m))

    def d(self, sigma):
        # print("STRIKE", self.strike_price)
        # print("PRICE", self.asset_price)
        d1 = 1 / (sigma * sqrt(self.T)) * (
                log(
                    self.asset_price/self.strike_price
                    ) + (self.rf_rate + sigma**2/2) * self.T
                )
        d2 = d1 - sigma * sqrt(self.T)
        return d1, d2

    def call_price_func(self, sigma, d1, d2):
        C = norm.cdf(d1) * self.asset_price - norm.cdf(d2) * \
            self.strike_price * exp(-self.rf_rate * self.T)
        return C

    def implied_volatility(self, tolerance=1e-4, epsilon=1, max_iter=1000):
        vol = 1
        count = 0
        while epsilon > tolerance:
            # Count how many iterations and make sure while loop doesn't run away
            count += 1
            if count >= max_iter:
                # print('Breaking on count')
                break

            #  Log the value previously calculated to computer percent change
            #  between iterations
            orig_vol = vol

            #  Calculate the vale of the call price
            d1, d2 = self.d(vol)
            function_value =\
                self.call_price_func(vol, d1, d2) - self.call_price

            #  Calculate vega, the derivative of the price with respect to
            #  volatility
            vega = self.asset_price * norm.pdf(d1) * sqrt(self.T)

            #  Update for value of the volatility
            vol = -function_value / vega + vol

            #  Check the percent change between current and last iteration
            epsilon = abs((vol - orig_vol) / orig_vol)
        return vol
