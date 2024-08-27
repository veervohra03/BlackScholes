import numpy as np
import scipy
import pandas as pd
import yfinance as yf
import datetime

class BlackScholes:
    def __init__(self, S: float, K: float, r: float, t: float, vol: float, q: float):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.vol = vol
        self.q = q

    def run(self):
        S = self.S 
        K = self.K
        r = self.r
        t = self.t / 365
        vol = self.vol
        q = self.q
        
        d1 = (np.log(S / K) + ((r - q + ((vol**2) / 2)) * t)) / (vol * np.sqrt(t))
        d2 = (np.log(S / K) + ((r - q - ((vol**2) / 2)) * t)) / (vol * np.sqrt(t))
    
        Nd1 = scipy.stats.norm.cdf(d1)
        Nd2 = scipy.stats.norm.cdf(d2)
        
        self.C = (S * np.exp(-q * t) * Nd1) - (K * np.exp(-r * t) * Nd2)
        self.P = (K * np.exp(-r * t)) + self.C - (np.exp(-q * t) * S)
        
        # Put-Call Parity
        # (np.exp(−r * T) * K) + self.C = (np.exp(−q * T) * S) + self.P

        # GREEKS

        self.c_theta = (-((S * vol * np.exp(-q * t)) / (2*np.sqrt(t)) * (1 / (np.sqrt(2 * np.pi))) * np.exp(-(d1 * d1) / 2))-(r * K * np.exp(-r * t) * Nd2) + (q * np.exp(-q * t) * S * Nd1)) / 365
        self.p_theta = (-((S * vol * np.exp(-q * t)) / (2*np.sqrt(t)) * (1 / (np.sqrt(2 * np.pi))) * np.exp(-(d1 * d1) / 2))+(r * K * np.exp(-r * t) * scipy.stats.norm.cdf(-d2)) - (q * np.exp(-q * t) * S * scipy.stats.norm.cdf(-d1))) / 365
        
        self.c_premium = np.exp(-q * t) * S * Nd1 - K * np.exp(-r * t) * scipy.stats.norm.cdf(d1 - vol * np.sqrt(t))
        self.p_premium = K * np.exp(-r * t) * scipy.stats.norm.cdf(-d2) - np.exp(-q * t) * S * scipy.stats.norm.cdf(-d1)
        
        self.c_delta = np.exp(-q * t) * Nd1
        self.p_delta = np.exp(-q * t) * (Nd1-1)
        
        self.gamma = (np.exp(-r*t)/(S*vol*np.sqrt(t)))*(1/(np.sqrt(2*np.pi)))*np.exp(-(d1*d1)/2)
        
        self.vega = ((1/100) * S * np.exp(-r * t)*np.sqrt(t))*(1/(np.sqrt(2*np.pi))*np.exp(-(d1*d1)/2))
        
        self.c_rho = (1/100) * K * t * np.exp(-r * t) * Nd2
        self.p_rho = (-1/100) * K * t * np.exp(-r * t) * scipy.stats.norm.cdf(-d2)
        
        return self.C, self.P, self.c_delta, self.p_delta, self.gamma, self.vega, self.c_theta, self.p_theta, self.c_rho, self.p_rho

def ticker_table(symbol, r, q):
    tk = yf.Ticker(symbol)
    exps = tk.options
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.concat([pd.DataFrame(opt.calls), pd.DataFrame(opt.puts)])
        opt['expirationDate'] = e
        options = pd.concat([options, opt], ignore_index=True)

    # add 1 day to adjust for error in yfinance that gives wrong expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['t'] = (options['expirationDate'] - datetime.datetime.today()).dt.days

    options['CALL?'] = options['contractSymbol'].str[4:].apply(lambda x: 'C' in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['theo'] = (options['bid'] + options['ask']) / 2
    
    # Drop unnecessary columns & add new columns
    options = options.drop(columns = ['contractSymbol', 'openInterest', 'contractSize', 'currency', 'change', 'percentChange', 'expirationDate', 'lastTradeDate', 'lastPrice'])
    options['delta'], options['gamma'], options['theta'], options['vega'], options['rho'] = 0.00, 0.00, 0.00, 0.00, 0.00
    
    data = tk.history()
    S = data['Close'].iloc[-1]
    
    for i in options.index:
        K = options.at[i, 'strike']
        t = options.at[i, 't']
        v = options.at[i, 'impliedVolatility']

        ticker_temp_bs_model = BlackScholes(S=S , K=K , r=r , t=t , vol=v , q=q)
        _, _, c_delta, p_delta, gamma, vega, c_theta, p_theta, c_rho, p_rho = ticker_temp_bs_model.run()

        options.at[i, 'vega'] = vega
        options.at[i, 'gamma'] = gamma

        if options.at[i, 'CALL?'] == True:
            options.at[i, 'delta'] = c_delta
            options.at[i, 'theta'] = c_theta
            options.at[i, 'rho'] = c_rho
        else:
            options.at[i, 'delta'] = p_delta
            options.at[i, 'theta'] = p_theta
            options.at[i, 'rho'] = p_rho
    
    options = options[['CALL?', 'volume', 't', 'bid', 'theo', 'ask', 'strike','impliedVolatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'inTheMoney']]

    return options, S

class BionomialLattice:
    def __init__(self, S: float, K: float, r: float, t: float, vol: float, steps: float):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
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
