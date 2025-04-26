import React, { useEffect, useState } from "react";
import axios from "axios";

function StockPrices() {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStockPrices = async () => {
      try {
        const stockSymbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]; // Add more symbols as needed
        const responses = await Promise.all(
          stockSymbols.map((symbol) =>
            axios.get(
              `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${process.env.FINNHUB_API_KEY}`
            )
          )
        );

        const stockData = responses.map((response, index) => ({
          symbol: stockSymbols[index],
          currentPrice: response.data.c,
          high: response.data.h,
          low: response.data.l,
        }));

        setStocks(stockData);
      } catch (error) {
        console.error("Error fetching stock prices:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStockPrices();
  }, []);

  return (
    <div className="p-5">
      <h1 className="text-2xl font-bold mb-4">Stock Prices</h1>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <table className="table-auto w-full border-collapse border border-gray-300">
          <thead>
            <tr>
              <th className="border border-gray-300 px-4 py-2">Symbol</th>
              <th className="border border-gray-300 px-4 py-2">Current Price</th>
              <th className="border border-gray-300 px-4 py-2">High</th>
              <th className="border border-gray-300 px-4 py-2">Low</th>
            </tr>
          </thead>
          <tbody>
            {stocks.map((stock) => (
              <tr key={stock.symbol}>
                <td className="border border-gray-300 px-4 py-2">{stock.symbol}</td>
                <td className="border border-gray-300 px-4 py-2">${stock.currentPrice}</td>
                <td className="border border-gray-300 px-4 py-2">${stock.high}</td>
                <td className="border border-gray-300 px-4 py-2">${stock.low}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default StockPrices;