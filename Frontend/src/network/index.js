
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
  const AdviceAI = async() => {
    const response = await axios.get("http://localhost:8000/advice")
    document.getElementById("getAdviceButton").addEventListener("click", async function () {
      const income = parseInt(document.getElementById("income").value.trim());
      const expenses = parseInt(document.getElementById("expenses").value.trim());
      const goals = document.getElementById("goals").value.trim().split(",").map(goal => goal.trim());

      if (!income || !expenses || goals.length === 0) {
          alert("Please fill in all fields!");
          return;
      }

      const userProfile = {
          income: income,
          expenses: expenses,
          goals: goals
      };

      try {
          const response = await fetch("/advice", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ userProfile })
          });

          if (!response.ok) {
              throw new Error("Failed to fetch advice");
          }

          const data = await response.json();
          document.getElementById("investmentAdvice").textContent = data.advice;
      } catch (error) {
          console.error("Error:", error);
          alert("Something went wrong. Check the console.");
      }
  });
  }

  ////////////////////////////////////////////////////////////////////
export { fetchAPI, saveNote, fetchNotes, deleteNote, editNote, AdviceAI };// funkciite koito se wry]at