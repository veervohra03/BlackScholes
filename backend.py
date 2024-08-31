import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, r, t, vol, q):
        self.S = np.array(S)
        self.K = np.array(K)
        self.r = np.array(r)
        self.t = np.array(t) / 365
        self.vol = np.array(vol)
        self.q = np.array(q)

    def run(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.vol ** 2) * self.t) / (self.vol * np.sqrt(self.t))
        d2 = d1 - self.vol * np.sqrt(self.t)

        call_price = (self.S * np.exp(-self.q * self.t) * norm.cdf(d1)) - (self.K * np.exp(-self.r * self.t) * norm.cdf(d2))
        put_price = (self.K * np.exp(-self.r * self.t) * norm.cdf(-d2)) - (self.S * np.exp(-self.q * self.t) * norm.cdf(-d1))
        call_delta = norm.cdf(d1) * np.exp(-self.q * self.t)
        put_delta = -norm.cdf(-d1) * np.exp(-self.q * self.t)
        gamma = norm.pdf(d1) / (self.S * self.vol * np.sqrt(self.t))
        vega = self.S * np.exp(-self.q * self.t) * norm.pdf(d1) * np.sqrt(self.t)
        call_theta = (-self.S * norm.pdf(d1) * self.vol / (2 * np.sqrt(self.t)) - self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(d2)) / 365
        put_theta = (-self.S * norm.pdf(d1) * self.vol / (2 * np.sqrt(self.t)) + self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(-d2)) / 365
        call_rho = self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(d2)
        put_rho = -self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(-d2)

        return call_price, put_price, call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho

class BionomialLattice:
    def __init__(self, S: float, K: float, r: float, t: float, vol: float, steps: float):
        self.S = S
        self.K = K
        self.r = r
        self.t = t / 365
        self.vol = vol
        self.steps = steps

    def binomial_call(self, S, K, r, t, vol, steps): 
        # Delta t, up and down factors
        dT = t / steps                             
        u = np.exp(vol * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(S * u**j * d**(steps - j)) for j in range(steps + 1)])

        a = np.exp(r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(S_T - K, 0.0)

        # Overriding option price 
        for i in range(steps - 1, -1, -1):
            V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]

    def binomial_put(self, S, K, r, t, vol, steps): 
        # Delta t, up and down factors
        dT = t / steps                             
        u = np.exp(vol * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(S * u**j * d**(steps - j)) for j in range(steps + 1)])

        a = np.exp(r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(K - S_T, 0.0)

        # Overriding option price 
        for i in range(steps - 1, -1, -1):
            V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1])

        return V[0]
    
    def calculate_prices(self):
        self.C = self.binomial_call(self.S, self.K, self.r, self.t, self.vol, self.steps)
        self.P = self.binomial_put(self.S, self.K, self.r, self.t, self.vol, self.steps)
        return self.C, self.P

def ticker(symbol, r, q):
    tk = yf.Ticker(symbol)
    exps = tk.options
    options = pd.DataFrame()
    
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.concat([pd.DataFrame(opt.calls), pd.DataFrame(opt.puts)])
        opt['expirationDate'] = e
        options = pd.concat([options, opt], ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + pd.DateOffset(days=1)
    options['t'] = (options['expirationDate'] - pd.Timestamp.today()).dt.days

    options['CALL?'] = options['contractSymbol'].str[4:].apply(lambda x: 'C' in x)
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['theo'] = (options['bid'] + options['ask']) / 2

    options = options.drop(columns=['contractSymbol', 'openInterest', 'contractSize', 'currency', 'change', 'percentChange', 'expirationDate', 'lastTradeDate', 'lastPrice'])
    options['delta'], options['gamma'], options['theta'], options['vega'], options['rho'] = 0.00, 0.00, 0.00, 0.00, 0.00

    data = tk.history()
    S = data['Close'].iloc[-1]

    bs_model = BlackScholes(S=S,
                            K=options['strike'],
                            r=r,
                            t=options['t'],
                            vol=options['impliedVolatility'],
                            q=q)
    
    _, __, call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho = bs_model.run()

    options['delta'] = np.where(options['CALL?'], call_delta, put_delta)
    options['theta'] = np.where(options['CALL?'], call_theta, put_theta)
    options['rho'] = np.where(options['CALL?'], call_rho, put_rho)
    options['gamma'] = gamma
    options['vega'] = vega

    options = options[['CALL?', 'volume', 't', 'bid', 'theo', 'ask', 'strike', 'impliedVolatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'inTheMoney']]

    # PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.suptitle('Volatility Smiles')
    unique_expirations = options['t'].unique()
    calls = options[options['CALL?'] == True]
    puts = options[options['CALL?'] == False]
    for t in unique_expirations:
        exp_calls = calls[calls['t'] == t]
        exp_puts = puts[puts['t'] == t]
        sns.lineplot(ax=ax1, data=exp_calls, x='strike', y='impliedVolatility', label=f'{int(t)}')
        sns.lineplot(ax=ax2, data=exp_puts, x='strike', y='impliedVolatility', label=f'{int(t)}')
    ax1.set_title("Calls")
    ax1.set(xlabel='Strike', ylabel='Implied Volatility')
    ax2.set_title("Puts")
    ax2.set(xlabel='Strike', ylabel='Implied Volatility')
    ax2.get_legend().remove()

    return options, S, fig
