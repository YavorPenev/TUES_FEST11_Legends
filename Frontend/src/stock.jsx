import React, { useEffect, useState } from "react";
import Header from './assets/header';
import Footer from './assets/footer';
import axios from "axios";

function StockPrices() {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);

  const FINNHUB_API_KEY = "cveirj9r01ql1jnbnjogcveirj9r01ql1jnbnjp0"; 

  useEffect(() => {
    const fetchAllStockSymbols = async () => {
      try {
  
        const response = await axios.get(
          `https://finnhub.io/api/v1/stock/symbol?exchange=US&token=${FINNHUB_API_KEY}`
        );

        const stockSymbols = response.data.map(stock => stock.symbol);


        const stockResponses = await Promise.all(
          stockSymbols.slice(0, 20).map((symbol) => 
            axios.get(
              `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}` 
            )
          )
        );


        const stockData = stockResponses.map((response, index) => ({
          symbol: stockSymbols[index],
          currentPrice: response.data.c,
          high: response.data.h,
          low: response.data.l,
          previousClose: response.data.pc,
        }));

        setStocks(stockData);
      } catch (error) {
        console.error("Error fetching stock prices:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchAllStockSymbols();
  }, [FINNHUB_API_KEY]);

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <div className="flex-grow p-5 max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 mt-20 text-center">Live Stock Prices for the day</h1>

        {loading ? (
          <div className="flex justify-center items-center">
            <p className="text-blue-500 animate-pulse">Loading stock data...</p>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-screen mb-10">
            <table className="table-auto w-full max-h-screen border-collapse border border-blue-300">
              <thead className="bg-blue-100">
                <tr>
                  <th className="border border-blue-300 px-6 py-3">Symbol</th>
                  <th className="border border-blue-300 px-6 py-3">Current Price</th>
                  <th className="border border-blue-300 px-6 py-3">High</th>
                  <th className="border border-blue-300 px-6 py-3">Low</th>
                </tr>
              </thead>
              <tbody>
                {stocks.map((stock) => {
                  const isUp = stock.currentPrice >= stock.previousClose;
                  return (
                    <tr
                      key={stock.symbol}
                      className="hover:bg-gray-100 transition-colors duration-300"
                    >
                      <td className="border border-blue-300 px-6 py-4 text-center font-semibold">
                        {stock.symbol}
                      </td>
                      <td
                        className={`border border-blue-300 px-6 py-4 text-center font-bold ${
                          isUp ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        ${stock.currentPrice.toFixed(2)}
                      </td>
                      <td className="border border-blue-300 px-6 py-4 text-center">
                        ${stock.high.toFixed(2)}
                      </td>
                      <td className="border border-blue-300 px-6 py-4 text-center">
                        ${stock.low.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
}

export default StockPrices;
