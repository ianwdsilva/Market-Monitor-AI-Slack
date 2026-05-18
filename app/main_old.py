if __name__ == "__main__":
    from app.config import settings

    print("--- Loaded Settings ---")
    print(f"Webhook URL: {settings.slack_webhook_url}")
    print(f"Watchlist Tickers: {settings.watchlist_tickers}")
    print("-----------------------")
