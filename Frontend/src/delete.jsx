
import React, { useState, useEffect } from 'react';
import './styles/delete.css';
import { fetchNotes, deleteNote } from './network/index'; // Импортиране на функциите

function Del() {
  const [notes, setNotes] = useState([]);
  const [titleToDelete, setTitleToDelete] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const getNotes = async () => {
      try {
        const data = await fetchNotes();
        setNotes(data);// Задава бележките в състоянието
      } catch (err) {
        console.error("Failed to fetch notes.");
      }
    };

    getNotes();
  }, []);

  useEffect(() => {
    if (message) {
      alert(message);
    }
  }, [message]); // ✅ Преместено извън return()

  const handleDelete = async () => {
    if (!titleToDelete) {
      setMessage('Please enter a title to delete.');
      return;
    }

    try {
      const response = await deleteNote(titleToDelete);
      setMessage(response.message);// Показва съобщение за успех
      setNotes(notes.filter((note) => note.title !== titleToDelete)); // Актуализира списъка
      setTitleToDelete('');
    } catch (error) {
      setMessage('Failed to delete the note.');
    }
  };

  const handleDeleteByClick = async (title) => {
    if (window.confirm("Are you sure you want to delete this note? This action cannot be undone.")) {
      try {
        const response = await deleteNote(title);
        setMessage(response.message);
        setNotes(notes.filter((note) => note.title !== title));
      } catch (error) {
        setMessage('Failed to delete the note.');
      }
    }
  };

  return (
    <div>
      <h1 className="text-green-500 delete-title"><b>Delete Notes</b></h1>
      <div className="delete-header"></div>
      <div className='delete-section'>
        <div className='delete-form'>
          <h2>Enter title to delete:</h2>
          <input
            type="text"
            value={titleToDelete}
            onChange={(e) => setTitleToDelete(e.target.value)}
          />
          <button onClick={handleDelete}><b>Delete</b></button>
        </div>
        <div className='delete-warning'>
          <h1 className='text-red-500 delete-warningt'>
            <b>Warning: Once you delete this note, it will be permanently removed and cannot be recovered. Please make sure you want to proceed before confirming!</b>
          </h1>
        </div>
      </div>

      <h2 className='delete-title1'><b>-- Choose notes to delete --</b></h2>
      <div className="delete-list">
        {notes.map((note) => (
          <div
            key={note.id}
            className="delete-item"
            onClick={() => handleDeleteByClick(note.title)}
             /*<div
            key={note.id}
            className="delete-item"
            onClick={() => alert("Are you sure you want to delete this note?")}
          >*/
          >
            <h3 className="delete-massage">{note.title}</h3>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Del;