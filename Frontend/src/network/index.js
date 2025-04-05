import axios from 'axios';

const fetchAPI = async (setArray) => {
  const response = await axios.get("http://localhost:8000/api");
  setArray(response.data.fruit);
  console.log(response.data.fruit);
};

//////////////////////////////////////////////////////////////////

const saveNote = async (title, body) => {
  try {
    const response = await axios.post("http://localhost:8000/add", {
      title,
      body,
    }, { withCredentials: true }); // <-- Added
    return response.data;
  } catch (error) {
    console.error("Error saving note:", error);
    throw error;
  }
};

/////////////////////////////////////////////////////////////////

const fetchNotes = async () => {
  try {
    const response = await axios.get("http://localhost:8000/notes", {
      withCredentials: true, // <-- Added
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching notes:", error);
    throw error;
  }
};

///////////////////////////////////////////////////////////////////

const deleteNote = async (title) => {
  try {
    const response = await axios.delete("http://localhost:8000/delete", {
      data: { title },
      withCredentials: true, // <-- Added
    });
    return response.data;
  } catch (error) {
    console.error("Error deleting note:", error);
    throw error;
  }
};

/////////////////////////////////////////////////////////////////////

const editNote = async (oldTitle, newTitle, newBody) => {
  try {
    const response = await axios.put("http://localhost:8000/edit", {
      oldTitle,
      newTitle,
      newBody,
    }, {
      withCredentials: true, // <-- Added
    });
    return response.data;
  } catch (error) {
    console.error("Error editing note:", error);
    throw error;
  }
};

///////////////////////////////////////////////////////////////////

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
      withCredentials: true, // <-- Added
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

/////////////////////////////////////////////////////////////////////

const login1 = async (username, password) => {
  try {
    const response = await axios.post("http://localhost:8000/login", {
      username,
      password,
    }, {
      withCredentials: true, // <-- Added
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

/////////////////////////////////////////////////////////////////////

const signup1 = async (username, email, password) => {
  try {
    const response = await axios.post("http://localhost:8000/signup", {
      username,
      email,
      password,
    }, {
      withCredentials: true, // <-- Added
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

/////////////////////////////////////////////////////////////////////

const Invest = async (investments, goals, setInvest) => {
  if (!investments || investments.length === 0) {
    alert("Please enter valid investments.");
    return;
  }

  if (!goals || goals.length === 0) {
    alert("Please enter valid goals.");
    return;
  }

  try {
    const response = await axios.post("http://localhost:8000/invest", {
      investments,
      goals
    }, {
      withCredentials: true, // <-- Added
    });

    if (response.status !== 200) {
      throw new Error(`Failed to fetch advice: ${response.statusText}`);
    }

    if (response.data && response.data.invest) {
      setInvest(response.data.invest);
    } else {
      throw new Error("No advice received from the backend.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert(`Something went wrong: ${error.message}`);
  }
};

/////////////////////////////////////////////////////////////////////

export {
  fetchAPI,
  saveNote,
  fetchNotes,
  deleteNote,
  editNote,
  Advice,
  login1,
  signup1,
  Invest
};
