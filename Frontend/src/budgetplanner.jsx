import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Header from './assets/header';
import Footer from './assets/footer';
import Notes from './assets/notes';
import Calculator from './assets/SimpCalc';



const BudgetPlanner = () => {
  const [income, setIncome] = useState('');
  const [expenses, setExpenses] = useState('');
  const [familySize, setFamilySize] = useState('');
  const [goals, setGoals] = useState('');
  const [budgetPlan, setBudgetPlan] = useState('');
  const [isClicked, setIsClicked] = useState(false);
  const planRef = useRef(null);

  const handleSavePlan = async () => {
    console.log("Save button clicked");
    if (!budgetPlan || budgetPlan.trim().length === 0) {
      alert("No budget plan to save!");
      return;
    }
  
    try {
      const response = await axios.post("http://localhost:8000/save-budget-plan", {
        plan: budgetPlan,
      });
  
      if (response.status === 201) {
        alert("Budget plan saved successfully!");
      } else {
        alert("Failed to save the budget plan.");
      }
    } catch (error) {
      console.error("Error saving budget plan:", error);
      alert("An error occurred while saving the budget plan.");
    }
  };

  useEffect(() => {
    document.body.style.cursor = isClicked ? "wait" : "default";
  }, [isClicked]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    if (name === 'income') setIncome(value);
    else if (name === 'expenses') setExpenses(value);
    else if (name === 'familySize') setFamilySize(value);
    else if (name === 'goals') setGoals(value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!income || !expenses || !familySize || !goals) {
      alert("Please fill in all fields.");
      return;
    }

    setIsClicked(true);

    const userData = {
      income: parseFloat(income),
      expenses: parseFloat(expenses),
      familySize: parseInt(familySize),
      goals,
    };

    try {
      const response = await axios.post('http://localhost:8000/budgetplanner', userData, {
        headers: { 'Content-Type': 'application/json' },
      });

      setBudgetPlan(response.data.plan);

      setTimeout(() => {
        planRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (error) {
      console.error("Error fetching the budget plan:", error);
      alert("There was an error generating the budget plan.");
    } finally {
      setIsClicked(false);
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gradient-to-r flex items-start justify-center p-6 mt-10">
        {/* Info Box */}
        <div className="bg-gray-100 shadow-md text-lg rounded-lg p-6 max-w-sm w-full mr-26 mt-34">
          <h2 className="text-xl font-bold text-gray-800 mb-4">How This Helps</h2>
          <p className="text-gray-700 text-lg">
            The Budget Planner helps you organize your finances based on:
          </p>
          <ul className="list-disc list-inside text-gray-700 text-lg mt-2 space-y-1">
            <li>Your income and expenses</li>
            <li>Family size</li>
            <li>Your specific financial goals</li>
          </ul>
          <p className="text-gray-700 text-lg mt-3">
            It generates a detailed plan to guide you in managing your finances and meeting your goals.
          </p>
        </div>

        {/* Form Box */}
        <div className="min-h-screen flex items-center justify-center p-6 mt-0">
          <form onSubmit={handleSubmit} className="bg-gray-100 shadow-lg rounded-lg p-8 max-w-lg w-full flex flex-col mt-3 space-y-4">
            <h1 className="text-4xl font-bold text-gray-800 mb-6 text-center">Budget Planner</h1>

            <div>
              <label htmlFor="income" className="block text-sm font-medium text-gray-700">Income ($):</label>
              <input
                type="number"
                id="income"
                name="income"
                placeholder="70000"
                value={income}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>

            <div>
              <label htmlFor="expenses" className="block text-sm font-medium text-gray-700">Expenses ($):</label>
              <input
                type="number"
                id="expenses"
                name="expenses"
                placeholder="3000"
                value={expenses}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>

            <div>
              <label htmlFor="familySize" className="block text-sm font-medium text-gray-700">Family Size:</label>
              <input
                type="number"
                id="familySize"
                name="familySize"
                placeholder="4"
                value={familySize}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-1"
              />
            </div>

            <div>
              <label htmlFor="goals" className="block text-sm font-medium text-gray-700">Financial Goals:</label>
              <textarea
                id="goals"
                name="goals"
                placeholder="Save for a house, pay off debt..."
                value={goals}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2"
              />
            </div>

            <button
              type="submit"
              disabled={isClicked}
              className={`bg-blue-800 text-white w-full font-bold py-2 px-4 rounded-md shadow-md flex items-center justify-center gap-2 mt-4 transition-transform hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 ${isClicked ? "bg-blue-700 cursor-wait" : "cursor-pointer"}`}
            >
              {isClicked ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  Generating Plan...
                </>
              ) : (
                "Generate Budget Plan"
              )}
            </button>
          </form>
        </div>
      </div>

      {/* Result Display */}
      <div className="mt-0" ref={planRef}>
        <h2 className="text-2xl font-bold text-gray-800 text-center">Your Budget Plan:</h2>
        <pre className="mt-2 bg-gray-100 p-4 rounded-xl text-gray-700 whitespace-pre-wrap font-sans mx-[17%] mb-10">
          {budgetPlan || "Your plan will appear here."}
        </pre>
        {budgetPlan && (
          <button
            onClick={handleSavePlan}
            className="mt-4 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition"
          >
            Save
          </button>
        )}
      </div>

      <Notes />
      <Calculator />
      <Footer />
    </>
  );
};

export default BudgetPlanner;
