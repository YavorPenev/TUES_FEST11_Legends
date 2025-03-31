
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

export { fetchAPI, saveNote, fetchNotes, deleteNote, editNote };// funkciite koito se wry]at