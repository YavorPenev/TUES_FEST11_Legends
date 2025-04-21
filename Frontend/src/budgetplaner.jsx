import React, { useState } from 'react';
import axios from 'axios';

const BudgetPlanner = () => {
  const [income, setIncome] = useState('');
  const [expenses, setExpenses] = useState('');
  const [familySize, setFamilySize] = useState('');
  const [goals, setGoals] = useState('');
  const [budgetPlan, setBudgetPlan] = useState('');
  const [isClicked, setIsClicked] = useState(false);
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    if (name === 'income') {
      setIncome(value);
    } else if (name === 'expenses') {
      setExpenses(value);
    } else if (name === 'familySize') {
      setFamilySize(value);
    } else if (name === 'goals') {
      setGoals(value);
    }
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
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setBudgetPlan(response.data.plan);
    } catch (error) {
      console.error("Error fetching the budget plan:", error);
      alert("There was an error generating the budget plan.");
    } finally {
      setIsClicked(false);
    }
  };
  
  return (
    <div>
      <h1>Budget Planner</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="income">Income:</label>
          <input
            type="number"
            id="income"
            name="income"
            value={income}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div>
          <label htmlFor="expenses">Expenses:</label>
          <input
            type="number"
            id="expenses"
            name="expenses"
            value={expenses}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div>
          <label htmlFor="familySize">Family Size:</label>
          <input
            type="number"
            id="familySize"
            name="familySize"
            value={familySize}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div>
          <label htmlFor="goals">Financial Goals:</label>
          <textarea
            id="goals"
            name="goals"
            value={goals}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div>
          <button type="submit" disabled={isClicked}>
            {isClicked ? "Generating Plan..." : "Generate Budget Plan"}
          </button>
        </div>
      </form>

      {budgetPlan && (
        <div>
          <h2>Generated Budget Plan:</h2>
          <pre>{budgetPlan}</pre>
        </div>
      )}
    </div>
  );
};

export default BudgetPlanner;
