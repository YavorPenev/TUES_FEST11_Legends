import React, { useState } from "react";
import { Advice } from "./network/index";

function AdviceAI() {
  const [income, setIncome] = useState("");
  const [expenses, setExpenses] = useState("");
  const [goals, setGoals] = useState("");
  const [advice, setAdvice] = useState("");

  const handleGetAdvice = () => {
    const goalsArray = goals.split(",").map((goal) => goal.trim());
    Advice(parseInt(income), parseInt(expenses), goalsArray, setAdvice);
  };

  return (
    <div>
      <div>
        <h2>Get Investment Advice</h2>
        <label htmlFor="income">Enter Your Income ($):</label>
        <input
          type="number"
          id="income"
          placeholder="70000"
          value={income}
          onChange={(e) => setIncome(e.target.value)}
        />
        <label htmlFor="expenses">Enter Your Monthly Expenses ($):</label>
        <input
          type="number"
          id="expenses"
          placeholder="2000"
          value={expenses}
          onChange={(e) => setExpenses(e.target.value)}
        />
        <label htmlFor="goals">Enter Your Investment Goals (comma separated):</label>
        <input
          type="text"
          id="goals"
          placeholder="long-term growth, low risk"
          value={goals}
          onChange={(e) => setGoals(e.target.value)}
        />

        <button type="button" onClick={handleGetAdvice}>
          Get Advice
        </button>
      </div>

      <h2 id="inv2">Investment Advice:</h2>
      <pre id="investmentAdvice">{advice}</pre>
    </div>
  );
}

export default AdviceAI;