from binance.client import Client
import pandas as pd
import os
import time
from dotenv import load_dotenv

os.makedirs("data/raw_data", exist_ok=True)

load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def date_to_milliseconds(date_str):
    return int(time.mktime(pd.to_datetime(date_str).timetuple()) * 1000)

def fetch_binance_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY, start_date="2019-01-01", end_date="2025-05-26"):
    start_ms = date_to_milliseconds(start_date)
    end_ms = date_to_milliseconds(end_date)

    all_data = []
    
    while start_ms < end_ms:
        klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_ms, limit=500)
        
        if not klines:
            break
        
        df = pd.DataFrame(klines, columns=["date", "open", "high", "low", "close", "volume", "close_time",
                                           "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        
        all_data.append(df)
        
        start_ms = int(klines[-1][0]) + 1

        time.sleep(0.2)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


if __name__ == "__main__":
    btc_data = fetch_binance_data(start_date="2025-07-14", end_date="2025-08-04")
    btc_data.to_csv("data/raw_data/btc_holdout_2.csv", index=False)