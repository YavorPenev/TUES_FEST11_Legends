import React, { useState } from "react";
import { Invest } from "./network/index";
import Header from './header';
import Footer from './footer';
import { Link } from "react-router";

function InvestAI() {
  const [symbols, setSymbols] = useState("");
  const [amount, setAmount] = useState("");
  const [goals, setGoals] = useState("");  // Added state for goals
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

    if (!goals || goals.trim().length === 0) {
      alert("Please enter your investment goals.");
      return;
    }

    const investments = symbolList.map((symbol) => ({
      symbol,
      amount: parseFloat(amount),
    }));

    const goalList = goals
      .split(",")
      .map((g) => g.trim())  // Split goals by commas and trim each goal
      .filter((g) => g !== "");

    if (goalList.length === 0) {
      alert("Please enter valid goals.");
      return;
    }

    // Call the Invest function with both investments and goals
    await Invest(investments, goalList, setResult);
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

        {/* New input field for goals */}
        <input
          type="text"
          placeholder="Enter your investment goals (e.g., Retirement, Buying a house)"
          value={goals}
          onChange={(e) => setGoals(e.target.value)}
          className="w-full max-w-xl px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-6"
        />

        <button
          onClick={handleInvestClick}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          Analyze My Investment
        </button>

        {result && (
          <div className="mt-10 w-full max-w-3xl bg-white p-6 rounded-lg shadow-md mb-[5rem]">
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Analysis Result:</h2>
            <p className="text-gray-700 whitespace-pre-line">{result}</p>
          </div>
        )}

      </div>

      <footer className="w-full flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center">
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
          <a
            href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" />
          </a>
          <p>Copyright @2025</p>
          <p>©legends Development Team</p>
        </div>
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
          <p>legends@gmail.com</p>
          <p>+39 06 6988 4857</p>
          <p>+39 04 5355 9832</p>
        </div>
        <div className="flex flex-col gap-1 items-center">
        <Link to="/investcalc" className="text-blue-100 hover:underline">Investment Calculator</Link>
          <Link to="/calcloan" className="text-blue-100 hover:underline">Loan Calculator</Link>
          <Link to="/CurrencyCalculator" className="text-blue-100 hover:underline">Currency Calculator</Link>
        </div>
        <div className="flex flex-col gap-1 items-start">
          <Link to="/articles" className="text-blue-100 hover:underline">Articles</Link>
          <Link to="/about" className="text-blue-100 hover:underline">About Us</Link>
          <Link to="/law" className="text-blue-100 hover:underline">Terms of Use</Link>
        </div>
      </footer>
    </div>
  );
}

export default InvestAI;



