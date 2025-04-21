import React, { useState, useEffect } from "react";
import { Advice } from "./network/index";
import Header from './assets/header';
import Footer from './assets/footer';
import Calculator from './assets/SimpCalc';
import Notes from './assets/notes';


function AdviceAI() {
  const [loading, setLoading] = useState(false); // Loading state for the button
  useEffect(() => {
    document.body.style.cursor = loading ? "wait" : "default";
  }, [loading]);

  const [income, setIncome] = useState("");
  const [expenses, setExpenses] = useState("");
  const [goals, setGoals] = useState("");
  const [advice, setAdvice] = useState("");

  const handleGetAdvice = async () => {
    const goalsArray = goals.split(",").map((goal) => goal.trim());

    if (!income || isNaN(income) || !expenses || isNaN(expenses)) {
      alert("Please enter valid income and expense values.");
      return;
    }

    if (!goalsArray.length || goalsArray[0] === "") {
      alert("Please enter at least one investment goal.");
      return;
    }

    setLoading(true); // Set loading to true

    await Advice(parseInt(income), parseInt(expenses), goalsArray, setAdvice);

    setLoading(false); // Set loading to false
  };

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gradient-to-r flex items-start justify-center p-6 mt-10">
        {/* Explanation Box */}
        <div className="bg-gray-100 shadow-md text-lg rounded-lg p-6 max-w-sm w-full mr-6 mt-65">
          <h2 className="text-xl font-bold text-gray-800 mb-4">What This Does</h2>
          <p className="text-gray-700 text-lg">
            This tool helps you receive investment advice based on:
          </p>
          <ul className="list-disc list-inside text-gray-700 text-lg mt-2 space-y-1">
            <li>Your annual income</li>
            <li>Your monthly expenses</li>
            <li>Your investment goals</li>
          </ul>
          <p className="text-gray-700 text-lg mt-3">
            After submitting the form, the system calculates your disposable income and provides tailored investment advice based on your goals.
          </p>
        </div>

        {/* Main Form */}
        <div className="bg-gray-100 shadow-lg rounded-lg p-8 max-w-lg w-full gap-1 flex flex-col mt-50 ml-20">
          <h1 className="text-4xl font-bold text-gray-800 mb-6 text-center">
            Investment Advice
          </h1>

          <div className="space-y-4">
            <div>
              <label
                htmlFor="income"
                className="block text-sm font-medium text-gray-700"
              >
                Enter Your Income ($):
              </label>
              <input
                type="number"
                id="income"
                placeholder="70000"
                value={income}
                onChange={(e) => setIncome(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>
            <div>
              <label
                htmlFor="expenses"
                className="block text-sm font-medium text-gray-700"
              >
                Enter Your Monthly Expenses ($):
              </label>
              <input
                type="number"
                id="expenses"
                placeholder="2000"
                value={expenses}
                onChange={(e) => setExpenses(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>
            <div>
              <label
                htmlFor="goals"
                className="block text-sm font-medium text-gray-700"
              >
                Enter Your Investment Goals (comma separated):
              </label>
              <input
                type="text"
                id="goals"
                placeholder="long-term growth, low risk"
                value={goals}
                onChange={(e) => setGoals(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>
            <button
              type="button"
              onClick={handleGetAdvice}
              disabled={loading}
              className={`bg-blue-800 text-white w-full font-bold py-2 px-4 rounded-md shadow-md flex items-center justify-center gap-2 mt-4 transition-transform hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 ${
                loading ? "bg-blue-700 cursor-wait" : "cursor-pointer"
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
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                  Thinking...
                </>
              ) : (
                "Get Advice"
              )}
            </button>
          </div>

          <div className="mt-6">
            <h2 className="text-lg font-semibold text-gray-800">Investment Advice:</h2>
            <pre
              id="investmentAdvice"
              className="mt-2 bg-gray-100 p-4 rounded-md text-gray-700 whitespace-pre-wrap font-sans"
            >
              {advice || "Your advice will appear here."}
            </pre>
          </div>
        </div>
      </div>
      <Notes />
      <Calculator />
      <Footer />
    </>
  );
}

export default AdviceAI;