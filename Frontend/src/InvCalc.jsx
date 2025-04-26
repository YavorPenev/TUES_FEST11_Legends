import React, { useState } from "react";
import Header from './assets/header';
import Footer from './assets/footer';
import Calculator from './assets/SimpCalc';
import Notes from './assets/notes';

function InvestCalc() {
  const [buyPrice, setBuyPrice] = useState("");
  const [currentPrice, setCurrentPrice] = useState("");
  const [dividend, setDividend] = useState("");
  const [shares, setShares] = useState("");
  const [result, setResult] = useState(null);

  const calculate = () => {
    const bp = parseFloat(buyPrice);
    const cp = parseFloat(currentPrice);
    const div = parseFloat(dividend);
    const sh = parseFloat(shares);

    if (!bp || !cp || !sh || isNaN(bp) || isNaN(cp) || isNaN(sh)) {
      alert("Please fill in all fields with valid values!");
      return;
    }

    const profit = (cp - bp) * sh;
    const totalDividend = div * sh;
    const totalReturnPercent = (((cp + div) / bp) - 1) * 100;

    setResult({
      profit: profit.toFixed(2),
      dividend: totalDividend.toFixed(2),
      returnPercent: totalReturnPercent.toFixed(2),
    });
  };

  return (
    <>
      <div className="bg-gray-100 flex flex-col min-h-screen relative ">
        <Header />

        {/* Main content container */}
        <div className="flex flex-col lg:flex-row justify-center items-start gap-8 mt-61 px-6 mb-40.5 ">
          {/* Explanation Box */}
          <div className="bg-white shadow-md rounded-lg p-6 w-2/3 max-w-xl  -mt-1 mr-20 ">
            <h2 className="text-xl font-bold text-gray-800 mb-4">How This Works</h2>
            <p className="text-gray-700 text-lg">
              This calculator helps you analyze your returns from stocks or ETFs based on:
            </p>
            <ul className="list-disc list-inside text-gray-700 text-lg mt-2 space-y-1">
              <li>Buy Price (the price you paid per share)</li>
              <li>Current Market Price</li>
              <li>Dividends earned per share</li>
              <li>Number of shares you own</li>
            </ul>
            <p className="text-gray-700 text-lg mt-3">
              It calculates your total profit, dividend income, and percentage return so you can better understand your investment performance.
            </p>
          </div>

          {/* Calculator Box */}
          <div className="w-full max-w-md p-4 rounded-2xl shadow-lg bg-white">
            <h1 className="text-2xl font-bold mb-4 text-center">Stock/ETF Returns 📊</h1>

            <div className="space-y-2">
              <input
                type="number"
                placeholder="Purchase Price"
                value={buyPrice}
                onChange={(e) => setBuyPrice(e.target.value)}
                className="border p-2 w-full rounded mb-5"
              />
              <input
                type="number"
                placeholder="Current Price"
                value={currentPrice}
                onChange={(e) => setCurrentPrice(e.target.value)}
                className="border p-2 w-full rounded mb-5"
              />
              <input
                type="number"
                placeholder="Dividend per Stock"
                value={dividend}
                onChange={(e) => setDividend(e.target.value)}
                className="border p-2 w-full rounded mb-5"
              />
              <input
                type="number"
                placeholder="Number of Stocks"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                className="border p-2 w-full rounded mb-6"
              />

              <button
                onClick={calculate}
                className="bg-blue-500 text-white w-full p-2 rounded hover:bg-blue-600"
              >
                Calculate 📈
              </button>
            </div>

            {result && (
              <div className="mt-4 space-y-2">
                <p>📊 Profit: <strong>{result.profit} BGN</strong></p>
                <p>💸 Dividend Income: <strong>{result.dividend} BGN</strong></p>
                <p>📈 Return: <strong>{result.returnPercent} %</strong></p>
              </div>
            )}
          </div>
        </div>

        <Notes />
        <Calculator />
        <Footer className="fixed bottom-0 left-0 w-full bg-blue-800 text-white" />
      </div>
    </>
  );
}

export default InvestCalc;