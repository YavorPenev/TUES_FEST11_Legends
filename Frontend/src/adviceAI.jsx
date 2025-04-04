import React from 'react'
import adviceAI from './network/index';

function AdviceAI() {
  return (
    <div className="container">
        <div className="cont-elem">
            <h2>Get Investment Advice</h2>
            <label htmlFor="income">Enter Your Income ($):</label>
            <input type="number" id="income" placeholder="70000" />
            <label htmlFor="expenses">Enter Your Monthly Expenses ($):</label>
            <input type="number" id="expenses" placeholder="2000" />
            <label htmlFor="goals">Enter Your Investment Goals (comma separated):</label>
            <input type="text" id="goals" placeholder="long-term growth, low risk" />

            <button type="button" id="getAdviceButton">Get Advice</button>
        </div>

        <h2 id="inv2">Investment Advice:</h2>
        <pre id="investmentAdvice"></pre>
    </div>
  );
}

export default AdviceAI;