import React, { useState } from "react";
import { Invest } from "./network/index";

function InvestAI() {
  const [symbols, setSymbols] = useState("");
  const [amount, setAmount] = useState("");
  const [result, setResult] = useState("");

  const handleInvestClick = async () => {
    const symbolList = symbols
      .split(",")
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s !== "");

    if (symbolList.length === 0 || !amount || isNaN(amount)) {
      alert("Please enter valid stock symbols and amount invested.");
      return;
    }

    const investments = symbolList.map((symbol) => ({
      symbol,
      amount: parseFloat(amount),
    }));

    await Invest(investments, setResult);
  };

  return (
    <div className="p-5 bg-gray-100 min-h-screen flex flex-col items-center">
      <h1 className="text-blue-600 text-4xl font-bold mb-10 text-center">
        -- InvestAI --
      </h1>

      <input
        type="text"
        placeholder="Enter stock symbols (e.g., AAPL, MSFT)"
        value={symbols}
        onChange={(e) => setSymbols(e.target.value)}
        className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-4"
      />

      <input
        type="number"
        placeholder="Enter amount invested per stock (e.g., 5000)"
        value={amount}
        onChange={(e) => setAmount(e.target.value)}
        className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-4"
      />

      <button
        onClick={handleInvestClick}
        className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
      >
        Analyze My Investment
      </button>

      {result && (
        <div className="mt-10 w-full max-w-3xl bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-2 text-gray-800">Analysis Result:</h2>
          <p className="text-gray-700 whitespace-pre-line">{result}</p>
        </div>
      )}
    </div>
  );
}

export default InvestAI;


