
import React, { useState, useEffect } from 'react';
import './styles/redact.css';
import { fetchNotes, editNote } from './network/index'; // Импортиране на функциите

function Redact() {
  const [notes, setNotes] = useState([]);
  const [selectedNote, setSelectedNote] = useState('');
  const [newTitle, setNewTitle] = useState('');
  const [newBody, setNewBody] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const getNotes = async () => {
      try {
        const data = await fetchNotes();
        setNotes(data); // Зареждаме бележките
      } catch (err) {
        console.error("Failed to fetch notes.");
      }
    };

    getNotes();
  }, []);

  // 🟢 Запълваме полетата при избор на бележка
  const handleSelectNote = (title) => {
    setSelectedNote(title);
    const note = notes.find((n) => n.title === title);
    if (note) {
      setNewTitle(note.title); // Автоматично попълва заглавието
      setNewBody(note.body);   // Автоматично попълва тялото
    }
  };

  const handleEdit = async () => {
    if (!selectedNote || !newTitle || !newBody) {
      setMessage('Please select a note and fill in all fields.');
      return;
    }

    try {
      const response = await editNote(selectedNote, newTitle, newBody);
      setMessage(response.message); // Показва съобщение за успех
      setNotes(
        notes.map((note) =>
          note.title === selectedNote ? { ...note, title: newTitle, body: newBody } : note
        )
      ); // Актуализира списъка
      setSelectedNote('');
      setNewTitle('');
      setNewBody('');
    } catch (error) {
      setMessage('Failed to edit the note.');
    }
  };

  return (
    <div>
      <h1 className="text-green-500 redact-title"><b>Redact Notes</b></h1>
      <div className="redact-header"></div>
      
      <h2>Select a note to edit:</h2>
      <select
        value={selectedNote}
        onChange={(e) => handleSelectNote(e.target.value)}
      >
        <option value="">-- Select a note --</option>
        {notes.map((note) => (
          <option key={note.id} value={note.title}>
            {note.title}
          </option>
        ))}
      </select>

      <h2>New Title:</h2>
      <input
        type="text"
        value={newTitle}
        onChange={(e) => setNewTitle(e.target.value)}
      />

      <h2>New Body:</h2>
      <textarea
        value={newBody}
        onChange={(e) => setNewBody(e.target.value)}
      ></textarea>

      <button onClick={handleEdit}><b>Edit</b></button>

      {message && <p>{message}</p>}
    </div>
  );
}

export default Redact;