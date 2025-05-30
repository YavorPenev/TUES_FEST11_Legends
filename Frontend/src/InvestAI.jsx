import React, { useState, useEffect, useRef } from "react";
import { Invest } from "./network/index";
import Header from './assets/header';
import Footer from './assets/footer';
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
  const resultRef = useRef(null);

  const handleSaveResult = async () => {
  try {
    const response = await fetch("http://localhost:8000/save-investment-advice", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ advice: result }),
    });

    if (response.ok) {
      alert("Result saved successfully!");
    } else {
      alert("Failed to save the result.");
    }
  } catch (error) {
    console.error("Error saving result:", error);
    alert("An error occurred while saving the result.");
  }
};

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

    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center">
      <Header />
      <div className="flex-grow p-5 bg-white flex flex-col lg:flex-row items-start justify-center gap-10 mt-14 pb-20">
     
        <div className="w-full lg:w-2/3 p-6 bg-gray-100 text-blue-900 rounded-2xl shadow-lg mt-35 ">
          <h2 className="text-xl font-bold mb-4"> How to Use the Stock Advisor</h2>
          <p className="text-sm leading-relaxed">
            <strong>Stock Symbols</strong><br />
            Enter one or more stock tickers like <code>AAPL</code>, <code>MSFT</code>, or <code>TSLA</code> separated by commas.
            <br /><br />
            <strong>Amount Invested</strong><br />
            Specify how much you're investing in <em>each</em> stock.
            <br /><br />
            <strong>Investment Goals</strong><br />
            Describe your financial objectives — for example, <em>retirement</em>, <em>saving for a house</em>, or <em>building wealth</em>.
            <br /><br />
            <b>The investment advice of this model is not very reliable. If you want more accurate advice, use our Invest Advisor.</b>
          </p>
        </div>
    
        <div className="w-full lg:w-1/2 flex flex-col items-center bg-gray-100 mt-34 px-14 py-5 rounded-2xl shadow-xl">
          <h1 className="text-blue-800 text-4xl font-extrabold mb-10 text-center mt-0">
            Stock Advisor
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
            className={`bg-blue-800 text-white px-6 py-2 rounded-lg transition flex items-center justify-center gap-2 hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 ${loading ? "bg-blue-700 cursor-wait" : "cursor-pointer"}`}
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
        </div>



      </div>
      {result && (
        <div
          ref={resultRef}
          className="mt-5 w-full max-w-3xl p-6 rounded-lg shadow-md mb-[5rem] bg-gray-100"
        >
          <h2 className="text-xl font-semibold mb-2 text-gray-800">
            Analysis Result:
          </h2>
          <p className="text-gray-700 whitespace-pre-line">{result}</p>
          <button
            onClick={handleSaveResult}
            className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
          >
            Save answear
          </button>
        </div>
      )}
      <Notes />
      <Calculator />
      <Footer />
    </div>
  );
}

export default InvestAI;



