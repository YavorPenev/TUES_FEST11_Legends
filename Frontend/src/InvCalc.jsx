
import React, { useState } from "react";
import Header from './assets/header';
import Footer from './assets/footer'; 

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

    if (!bp || !cp || !div || !sh || isNaN(bp) || isNaN(cp) || isNaN(div) || isNaN(sh)) {
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
      <Header /> 
      <div className="mt-35 mb-10 max-w-md mx-auto p-4 rounded-2xl shadow-lg bg-white">
        <h1 className="text-2xl font-bold mb-4 text-center">Stock/ETF Returns 📊</h1>

        <div className="space-y-2">
          <input
            type="number"
            placeholder="Purchase Price"
            value={buyPrice}
            onChange={(e) => setBuyPrice(e.target.value)}
            className="border p-2 w-full rounded"
          />
          <input
            type="number"
            placeholder="Current Price"
            value={currentPrice}
            onChange={(e) => setCurrentPrice(e.target.value)}
            className="border p-2 w-full rounded"
          />
          <input
            type="number"
            placeholder="Dividend per Stock"
            value={dividend}
            onChange={(e) => setDividend(e.target.value)}
            className="border p-2 w-full rounded"
          />
          <input
            type="number"
            placeholder="Number of Stocks"
            value={shares}
            onChange={(e) => setShares(e.target.value)}
            className="border p-2 w-full rounded"
          />

          <button
            onClick={calculate}
            className="bg-green-500 text-white w-full p-2 rounded hover:bg-green-600"
          >
            Calculate 📈
          </button>
        </div>

        {result && (
          <div className="mt-4 space-y-2 text-lg">
            <p>📊 Profit: <strong>{result.profit} BGN</strong></p>
            <p>💸 Dividend Income: <strong>{result.dividend} BGN</strong></p>
            <p>📈 Return: <strong>{result.returnPercent} %</strong></p>
          </div>
        )}
      </div>
      <Footer /> 
    </>
  );
}

export default InvestCalc;