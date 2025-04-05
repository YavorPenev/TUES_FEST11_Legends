import React, { useState } from 'react';

function Calc(loan, interest, months) {
  const minterest = interest / 100 / 12;
  const payment = (loan * minterest * (1 + minterest) ** months) / ((1 + minterest) ** months - 1);
  return payment;
}

function LoanCalc() {
  const [loan, setLoan] = useState('');
  const [interest, setInterest] = useState('');
  const [months, setMonths] = useState('');
  const [payment, setPayment] = useState(0);
  const [interestPaid, setInterestPaid] = useState(0);

  const handleCalculate = () => {
    const numLoan = Number(loan);
    const numInterest = Number(interest);
    const numMonths = Number(months);

    if (numLoan > 0 && numInterest > 0 && numMonths > 0) {
      const calculatedPayment = Calc(numLoan, numInterest, numMonths);
      setPayment(calculatedPayment.toFixed(2));

      const calculatedInterestPaid = (calculatedPayment * numMonths) - numLoan;
      setInterestPaid(calculatedInterestPaid.toFixed(2));
    } else {
      setPayment(0);
      setInterestPaid(0);
    }
  };

  return (
    <div className="mt-36 h-80 mb-48">
      <h1 className="text-blue-900 text-4xl font-bold mb-10 text-center">
        Advanced Loan Calculator
      </h1>

      <div className='flex justify-evenly flex-col absolute left-[15%] w-[30%] mr-[10%] h-80 justify-items-center items-center bg-blue-800 rounded-3xl'>
        <h2 className='p-4 text-blue-900 font-bold rounded-2xl bg-blue-50 w-[40%] hover:scale-110 transition-transform duration-200'>
          Enter your loan details:
        </h2>

        <input
          className=' text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform duration-100'
          type="number"
          value={loan}
          onChange={(e) => setLoan(e.target.value)}
          placeholder="Loan Amount..."
        />

        <input
          className=' text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform duration-100'
          type="number"
          value={interest}
          onChange={(e) => setInterest(e.target.value)}
          placeholder="Yearly Interest Rate..."
        />

        <input
          className='text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform duration-100'
          type="number"
          value={months}
          onChange={(e) => setMonths(e.target.value)}
          placeholder="Months of Payment..."
        />

        <button
          className='mb-3.5 mt-3.5 pr-5 pl-5 text-2xl font-bold pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-150 transition-transform duration-300'
          onClick={handleCalculate}
        >
          Calculate
        </button>
      </div>

      <div className='flex justify-evenly flex-col absolute right-[15%] w-[30%] ml-[10%] h-80 justify-items-center items-center bg-blue-800 rounded-3xl'>
        <div className='bg-blue-50 rounded-2xl p-8 text-blue-900 w-[50%] font-bold hover:scale-110 transition-transform duration-200'>
          <h3>Your monthly payment:</h3>
          <h4 className='text-2xl'>{payment}</h4>
        </div>
        <div className='bg-blue-50 rounded-2xl p-8 text-blue-900 w-[50%] font-bold hover:scale-110 transition-transform duration-200'>
          <h3>Total interest paid:</h3>
          <h4 className='text-2xl'>{interestPaid}</h4>
        </div>
      </div>
    </div>
  );
}

export default LoanCalc;
