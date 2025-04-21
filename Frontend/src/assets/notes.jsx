import { saveNote, fetchNotes } from "../network/index";
import React, { useState, useEffect, useRef } from "react";
import Draggable from "react-draggable";
import { Link } from "react-router";

function Notes() {
  const [isVisible, setIsVisible] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [notes, setNotes] = useState([]); 
  const nodeRef = useRef(null);


  useEffect(() => {
    const loadNotes = async () => {
      try {
        const data = await fetchNotes();
        setNotes(data);
      } catch (error) {
        console.error("Failed to fetch notes:", error);
      }
    };

    loadNotes();
  }, []);

  const handleSave = async () => {
    if (title.trim() === "" || content.trim() === "") {
      alert("Please fill in both the title and the content.");
      return;
    }

    try {
      await saveNote(title, content);
      alert("Note saved successfully!");
      setTitle("");
      setContent("");
      setIsVisible(false);

      const data = await fetchNotes();
      setNotes(data);
    } catch (error) {
      alert("Access denied. Please log in to access this resource.");
    }
  };

  return (
    <>
   
      {!isVisible && (
        <button
          onClick={() => setIsVisible(true)}
          className="fixed bottom-4 right-4 bg-blue-700 text-white px-4 py-2 rounded-lg shadow-lg hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 font-bold"
        >
          Take Note
        </button>
      )}


      {isVisible && (
        <Draggable nodeRef={nodeRef}>
          <div
            ref={nodeRef}
            className="fixed bottom-20 right-4 bg-blue-100 p-4 rounded-lg shadow-2xl w-96 border-2 border-blue-500 cursor-grab active:cursor-grabbing"
          >
            <div className="mb-2">
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter title"
                className="w-full p-2 border border-blue-300 rounded text-lg font-semibold bg-blue-50 text-blue-900"
              />
            </div>
            <div className="mb-2">
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Enter your note here..."
                className="w-full p-2 border border-blue-500 rounded bg-blue-50 text-blue-900 h-32"
              />
            </div>
            <div className="flex justify-between">
            <button
                onClick={() => setIsVisible(false)} 
                className="bg-red-500 text-white px-4 py-2 rounded-lg transition-transform hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 font-bold"
              >
                Close
              </button>
              <div>
              <Link
                to="/profile"
                className="inline-block bg-green-600 text-white px-4 mr-4 py-2 transition-transform rounded-lg hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 font-bold"
              >
                
                View notes
              </Link>
                
              <button
                onClick={handleSave}
                className="bg-blue-800 text-white px-4 py-2 transition-transform rounded-lg hover:scale-110 hover:duration-200 active:scale-90 active:duration-50 font-bold"
              >
                Save
              </button>
              </div>
           
            </div>
          </div>
        </Draggable>
      )}

    </>
  );
}

export default Notes;