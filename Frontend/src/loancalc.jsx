import React, { useState } from 'react';
import Header from './assets/header'
import Footer from './assets/footer';
import { Link } from 'react-router';
import Calculator from './assets/SimpCalc';
import Notes from './assets/notes';


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
    <>

      <Header />
      <div className='absolute left-[2%] top-[42.5%]  w-[28%] min-h-max bg-blue-800 bg-opacity-90 text-blue-900 p-6 rounded-2xl shadow-lg '>
        <div className='  bg-white rounded-2xl p-4'>
          <h2 className='text-xl font-bold mb-1'>How this calculator works:</h2>
          <p className=' leading-relaxed -mb-2'>
            This loan calculator helps you estimate your monthly payment and total interest paid over the life of your loan.
            <br /><br />
            <strong>Loan Amount:</strong> The total amount you’re borrowing.
            <br />
            <strong>Interest Rate:</strong> Annual percentage rate (APR) of the loan.
            <br />
            <strong>Months of Payment:</strong> How many months you’ll take to repay the loan.
            <br /><br />
            Press <strong>Calculate</strong> to get your results!
          </p>
        </div>
      </div>
      <div className="mt-26 h-110 mb-40">
        <h1 className="text-blue-900 text-4xl font-bold mb-5 text-center mt-46">
          Advanced Loan Calculator
        </h1>

        <div className='flex justify-evenly flex-col absolute left-[34%] w-[30%] mr-[10%]  h-96 justify-items-center items-center bg-blue-800 rounded-3xl mt-10'>
          <h2 className='p-4 text-blue-900 font-bold rounded-2xl bg-blue-50 w-[40%] hover:scale-110 transition-transform duration-200'>
            Enter your loan details:
          </h2>

          <input
            className=' text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform hover:duration-200 active:scale-100 active:duration-50'
            type="number"
            value={loan}
            onChange={(e) => setLoan(e.target.value)}
            placeholder="Loan Amount"
          />

          <input
            className=' text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform hover:duration-200 active:scale-100 active:duration-20'
            type="number"
            value={interest}
            onChange={(e) => setInterest(e.target.value)}
            placeholder="Yearly Interest Rate (%)"
          />

          <input
            className='text-center pr-8 pl-8 pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-105 transition-transform hover:duration-200 active:scale-100 active:duration-20'
            type="number"
            value={months}
            onChange={(e) => setMonths(e.target.value)}
            placeholder="Months of Payment"
          />

          <button
            className='mb-3.5 mt-3.5 pr-5 pl-5 text-2xl font-bold pt-1 pb-1 rounded-xl bg-blue-50 text-blue-900 hover:scale-150 transition-transform hover:duration-300 active:scale-120 active:duration-50'
            onClick={handleCalculate}
          >
            Calculate
          </button>
        </div>

        <div className='pb-10 pt-10 flex justify-evenly flex-col absolute right-[2%] w-[30%] ml-[10%] h-96 justify-items-center items-center bg-blue-800 rounded-3xl mt-10 '>
          <div className='bg-blue-50 rounded-2xl p-8 text-blue-900 w-[50%] font-bold hover:scale-110 transition-transform hover:duration-200'>
            <h3>Your monthly payment:</h3>
            <h4 className='text-2xl'>{payment}</h4>
          </div>
          <div className='bg-blue-50 rounded-2xl p-8 text-blue-900 w-[50%] font-bold hover:scale-110 transition-transform hover:duration-200'>
            <h3>Total interest paid:</h3>
            <h4 className='text-2xl'>{interestPaid}</h4>
          </div>
        </div>
      </div>
      <Notes />
      <Calculator />
      <Footer />


    </>
  );
}

export default LoanCalc;
