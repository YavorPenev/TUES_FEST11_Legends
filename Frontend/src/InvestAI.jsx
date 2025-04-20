import React, { useState, useEffect } from "react";
import { Invest } from "./network/index";
import Header from './assets/header';
import Footer from './assets/footer';
import { Link } from "react-router";
import Calculator from './assets/SimpCalc';
import Notes from './assets/notes';


function InvestAI() {

  const [loading, setLoading] = useState(false); // Loading state for the button
  useEffect(() => {
    document.body.style.cursor = loading ? 'wait' : 'default';
  }, [loading]);

  const [symbols, setSymbols] = useState("");
  const [amount, setAmount] = useState("");
  const [goals, setGoals] = useState("");
  const [result, setResult] = useState("");

  const handleInvestClick = async () => {
    setLoading(true); // start loading

    const symbolList = symbols
      .split(",")
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s !== "");

    if (symbolList.length === 0 || !amount || isNaN(amount)) {
      alert("Please enter valid stock symbols and amount invested.");
      setLoading(false);
      return;
    }

    if (!goals || goals.trim().length === 0) {
      alert("Please enter your investment goals.");
      setLoading(false);
      return;
    }

    const investments = symbolList.map((symbol) => ({
      symbol,
      amount: parseFloat(amount),
    }));

    const goalList = goals
      .split(",")
      .map((g) => g.trim())
      .filter((g) => g !== "");

    if (goalList.length === 0) {
      alert("Please enter valid goals.");
      setLoading(false);
      return;
    }

    await Invest(investments, goalList, setResult);

    setLoading(false); // stop loading
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex-grow p-5 bg-gray-100 flex flex-col items-center mt-24">

        <h1 className="text-blue-800 text-4xl font-bold mb-10 text-center mt-30">
          -- Stock Advisor --
        </h1>

        <input
          type="text"
          placeholder="Enter stock symbols (e.g., AAPL, MSFT)"
          value={symbols}
          onChange={(e) => setSymbols(e.target.value)}
          className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-6"
        />

        <input
          type="number"
          placeholder="Enter amount invested per stock (e.g., 5000)"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-6"
        />


        <input
          type="text"
          placeholder="Enter your investment goals (e.g., Retirement, Buying a house)"
          value={goals}
          onChange={(e) => setGoals(e.target.value)}
          className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-6"
        />

        <button
          onClick={handleInvestClick}
          disabled={loading}
          className={`bg-blue-800 text-white px-6 py-2 rounded-lg transition flex items-center justify-center gap-2 hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 ${loading ? "bg-blue-700 cursor-wait" : "cursor-pointer"
            }`}
        >
          {loading ? (
            <>
              <svg
                className="animate-spin h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
              </svg>
              Analyzing...
            </>
          ) : (
            "Analyze My Investment"
          )}
        </button>

        {result && (
          <div className="mt-10 w-full max-w-3xl bg-white p-6 rounded-lg shadow-md mb-[5rem]">
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Analysis Result:</h2>
            <p className="text-gray-700 whitespace-pre-line">{result}</p>
          </div>
        )}

      </div>
      <Notes />
      <Calculator />
      <Footer />
    </div>
  );
}

export default InvestAI;



