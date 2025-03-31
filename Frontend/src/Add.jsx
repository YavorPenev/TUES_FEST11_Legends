
import { useState, useEffect } from 'react';
import './styles/Add.css';
import { saveNote } from './network/index'; // import index.js

function Add() {
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [message, setMessage] = useState('');

  const handleSave = async () => {
    if (!title || !body) {
      setMessage('Title and text are required!');
      return;
    }

    try {
      const response = await saveNote(title, body);
      setMessage(response.message); // Показва съобщение за успех
      setTitle(''); // Изчиства полетата
      setBody('');
    } catch (error) {
      setMessage('Failed to save the note.');
    }
  };

  // alert-а само когато message се промени
  useEffect(() => {
    if (message) {
      alert(message);
    }
  }, [message]);

  return (
    <div>
      <h1 className="text-green-500 add-title"><b>Add Note</b></h1>
      <div className='add-header'></div>
      <div className='add-form'>
        <h2 className='add-titles'>Enter title:</h2>
        <input 
          placeholder='Enter the title of your note.'
          className='add-input'
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <h2 className='add-titles'>Enter text:</h2>
        <textarea
          placeholder='Enter the body of your note.'
          className='add-input'
          value={body}
          onChange={(e) => setBody(e.target.value)}
        ></textarea>
        <button onClick={handleSave}><b>Save</b></button>
      </div>
    </div>
  );
}

export default Add;