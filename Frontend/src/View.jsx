import React, { useState, useEffect } from 'react';
import './styles/View.css';
import { fetchNotes } from './network/index'; // Импортиране на функцията за извличане на бележки

function View() {
  const [notes, setNotes] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const getNotes = async () => {
      try {
        const data = await fetchNotes();
        setNotes(data); // Задава бележките в състоянието
      } catch (err) {
        setError("Failed to fetch notes.");
      }
    };

    getNotes();
  }, []);

  return (
    <div>
      <h1 className="text-green-500 view-title"><b>Your Notes</b></h1>
      <div className="view-header"></div>
      {error && <p className="text-red-500">{error}</p>}
      <div className="notes-list">
        {notes.length > 0 ? (
          notes.map((note) => (
            <div key={note.id} className="note-item">
              <h2 className="note-title">{note.title}</h2>
              <div className="view-header1"></div>
              <p className="note-body">{note.body}</p>
            </div>
          ))
        ) : (
          <p>No notes available.</p>
        )}
      </div>
    </div>
  );
}

export default View;