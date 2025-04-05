import axios from 'axios'


const fetchAPI = async (setArray) => {
    const response = await axios.get("http://localhost:8000/api");
    setArray(response.data.fruit);
    console.log(response.data.fruit);
  };

  //////////////////////////////////////////////////////////////

  const saveNote = async (title, body) => {
    try {
      const response = await axios.post("http://localhost:8000/add", {
        title,
        body,
      });
      return response.data; // Връща съобщение за успех или грешка
    } catch (error) {
      console.error("Error saving note:", error);
      throw error; // Прехвърля грешката за обработка в Add.jsx
    }
  };

  /////////////////////////////////////////////////////////////////

  const fetchNotes = async () => {
    try {
      const response = await axios.get("http://localhost:8000/notes");
      return response.data; // Връща списъка с бележки
    } catch (error) {
      console.error("Error fetching notes:", error);
      throw error; // Прехвърля грешката за обработка в View.jsx
    }
  };

  ///////////////////////////////////////////////////////////////////
  
  const deleteNote = async (title) => {
    try {
      const response = await axios.delete("http://localhost:8000/delete", {
        data: { title },
      });
      return response.data; // Връща съобщение за успех
    } catch (error) {
      console.error("Error deleting note:", error);
      throw error; // Прехвърля грешката за обработка в delete.jsx
    }
  };
  
  /////////////////////////////////////////////////////////////////////

  const editNote = async (oldTitle, newTitle, newBody) => {
    try {
      const response = await axios.put("http://localhost:8000/edit", {
        oldTitle, // Използваме oldTitle, както е в бекенда
        newTitle,
        newBody,
      });
      return response.data;
    } catch (error) {
      console.error("Error editing note:", error);
      throw error;
    }
  };
  ////////////////////////////////////////////////////////////////////
  const Advice = async (income, expenses, goals, setAdvice) => {
    if (!income || isNaN(income) || !expenses || isNaN(expenses) || goals.length === 0) {
      alert("Please fill in all fields with valid values!");
      return;
    }
  
    const userProfile = {
      income: parseInt(income),
      expenses: parseInt(expenses),
      goals: goals,
    };
  
    try {
      const response = await axios.post("http://localhost:8000/advice", userProfile, {
        headers: { "Content-Type": "application/json" },
      });
  
      if (response.status !== 200) {
        throw new Error(`Failed to fetch advice: ${response.statusText}`);
      }
  
      setAdvice(response.data.advice);
    } catch (error) {
      console.error("Error:", error);
      alert(`Something went wrong: ${error.message}`);
    }
  };
  ////////////////////////////////////////////////////////////////////

  const login1 = async (username, password) => {
    try {
      const response = await axios.post("http://localhost:8000/login", {
        username,
        password,
      });
  
      if (response.status !== 200) {
        throw new Error(`Login failed: ${response.data.error}`);
      }
  
      return response.data.user;
    } catch (error) {
      console.error("Error during login:", error);
      throw error; 
    }
  };

  ////////////////////////////////////////////////////////////////////
  const signup1 = async (username, email, password) => {
    try {
      const response = await axios.post("http://localhost:8000/signup", {
        username,
        email,
        password,
      });
  
      if (response.status !== 201) {
        throw new Error(`Signup failed: ${response.data.error}`);
      }

      return response.data.message;
    } catch (error) {
      console.error("Error during signup:", error);
      throw error;
    }
  };
  ///////////////////////////////////////////////////////////////////
// This function expects an array of investments and a setter function for the result
const Invest = async (investments, setInvest) => {
  if (!investments || investments.length === 0) {
    alert("Please enter valid investments.");
    return;
  }

  try {
    // Make a POST request to the backend with the investments data
    const response = await axios.post("http://localhost:8000/invest", {
      investments,
    });

    if (response.status !== 200) {
      throw new Error(`Failed to fetch advice: ${response.statusText}`);
    }

    // Check if the response contains valid data and set it to the state
    if (response.data && response.data.invest) {
      setInvest(response.data.invest); // Set the response data to the result state
    } else {
      throw new Error("No advice received from the backend.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert(`Something went wrong: ${error.message}`);
  }
};

  
  ////////////////////////////////////////////////////////////////////

export { fetchAPI, saveNote, fetchNotes, deleteNote, editNote, Advice, login1, signup1, Invest };// funkciite koito se wry]at