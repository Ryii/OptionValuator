class BlackScholes():
    def __init__(self, curPrice: int, strikePrice: int, timeUntilExpiration: int, volatility: int, interestRate: int):
        self.curPrice = curPrice
        self.strikePrice = strikePrice
        self.timeUntilExpiration = timeUntilExpiration
        self.volatility = volatility
        self.interestRate = interestRate

    def __repr__(self):
        print(f"Cur Price: {self.curPrice}")
        print(f"Strike Price: {self.strikePrice}")
        print(f"Time Until Expiration: {self.timeUntilExpiration}")
        print(f"Volatility: {self.volatility}")
        print(f"Interest Rate: {self.interestRate}")